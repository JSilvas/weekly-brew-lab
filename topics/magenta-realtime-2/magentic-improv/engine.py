"""
ImprovEngine — Magenta RT2 with real-time audio-to-MIDI conditioning.

Audio pipeline:
    mic input (sounddevice) → pitch detection (audio_to_midi) →
    NoteStateTracker → RT2 notes conditioning

Style pipeline:
    text prompts + weights → MusicCoCa embedding → RT2 style conditioning

Both pipelines run in parallel:
  - Mic callback fires every ~40 ms (BLOCKSIZE / SAMPLE_RATE) on the sounddevice thread.
    Pitch detection runs inline in the callback and calls _update_notes().

  - RT2 generate() runs on a dedicated MLX thread (MLX GPU streams are thread-local;
    the model must be loaded and used on the same thread).

MLX thread constraint: _do_load() and _gen_loop() are always submitted to self._mlx.
All other methods are safe to call from any thread.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sounddevice as sd

from audio_to_midi import NoteStateTracker, detect_pitch, hz_to_midi, midi_to_name

MAGENTA_HOME = os.environ.get(
    "MAGENTA_HOME",
    str(__import__("pathlib").Path.home() / "Documents" / "Magenta" / "magenta-rt-v2"),
)

SAMPLE_RATE   = 48_000
CHANNELS      = 2
FRAMES_CHUNK  = 50        # 2 s per generate() call
BLOCKSIZE     = 1_920     # 40 ms — one RT2 codec frame
RING_MAXCHUNKS = 8        # ~16 s ceiling

# Accumulate N sounddevice blocks before running pitch detection.
# 4096 samples ≈ 85 ms — enough for ~2 periods of 80 Hz (lowest tracked pitch).
PITCH_ACCUM = 4096


class AudioRingBuffer:
    def __init__(self):
        self._q: deque[np.ndarray] = deque()
        self._partial: np.ndarray | None = None
        self._offset = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray):
        with self._lock:
            if len(self._q) < RING_MAXCHUNKS:
                self._q.append(data.astype(np.float32))

    def read(self, n_frames: int) -> np.ndarray:
        out = np.zeros((n_frames, CHANNELS), dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < n_frames:
                if self._partial is not None:
                    avail = len(self._partial) - self._offset
                    take  = min(avail, n_frames - filled)
                    out[filled:filled + take] = self._partial[self._offset:self._offset + take]
                    filled       += take
                    self._offset += take
                    if self._offset >= len(self._partial):
                        self._partial = None
                        self._offset  = 0
                elif self._q:
                    self._partial = self._q.popleft()
                    self._offset  = 0
                else:
                    break
        return out

    @property
    def buffered_seconds(self) -> float:
        with self._lock:
            n = sum(len(c) for c in self._q)
            if self._partial is not None:
                n += len(self._partial) - self._offset
        return n / SAMPLE_RATE


class ImprovEngine:
    def __init__(self):
        self._mlx = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._mrt = None
        self._loaded = False
        self._playing = False

        # ── Style ──────────────────────────────────────────────────────────────
        self._style_lock = threading.Lock()
        self._current_embed: np.ndarray | None = None
        self._embed_cache: dict[str, np.ndarray] = {}
        self._ramp_thread: threading.Thread | None = None
        self._ramp_cancel = False

        # ── Output ─────────────────────────────────────────────────────────────
        self._ring = AudioRingBuffer()
        self._volume = 0.7
        self._sd_out: sd.OutputStream | None = None
        self._out_device_name: str = ""          # tracks current default/pinned device
        self._device_monitor: threading.Thread | None = None
        # When set, output is pinned to this device index instead of the system default.
        # Use this in System Audio mode to prevent Magenta's output from feeding back
        # into the loopback input device.
        self.preferred_out_device: int | None = None

        # ── Mic + pitch ────────────────────────────────────────────────────────
        self._mic_stream: sd.InputStream | None = None
        self._pitch_buf = np.zeros(PITCH_ACCUM, dtype=np.float32)
        self._mic_level = 0.0
        self._note_tracker = NoteStateTracker()
        self._notes_lock = threading.Lock()
        self._midi_notes: list[int] | None = None   # None = masked (unconditional)

        # Live display state (written by mic callback, read by UI timer)
        self._detected_freq: float | None = None
        self._detected_midi: int | None = None
        self._detected_confidence = 0.0

        # Config (safe to write from any thread; read on next gen frame)
        self.confidence_threshold = 0.4
        self.mic_enabled = False
        self._active_input_name: str = ""

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, model_size: str = "mrt2_small") -> str:
        try:
            return self._mlx.submit(self._do_load, model_size).result(timeout=120)
        except Exception as e:
            return f"✗ Load failed: {e}"

    def _do_load(self, model_size: str) -> str:
        from magenta_rt import paths, MagentaRT2Mlxfn
        paths.set_magenta_home(MAGENTA_HOME)
        self._mrt = MagentaRT2Mlxfn(size=model_size)
        self._loaded = True
        self._embed_cache.clear()
        # Seed a default style immediately so the gen loop can produce audio
        # as soon as Play is pressed, without waiting for the canvas bridge.
        # The style timer will update this once the canvas bridge fires.
        default = np.array(self._mrt.embed_style("Jazz Piano Trio"))
        norm = np.linalg.norm(default)
        with self._style_lock:
            self._current_embed = default / norm if norm > 1e-8 else default
        return f"✓ {model_size} loaded"

    # ── Style embedding ────────────────────────────────────────────────────────
    # embed_style uses TFLite/CPU — safe to call from any thread.

    def _embed(self, text: str) -> np.ndarray | None:
        text = text.strip()
        if not text or self._mrt is None:
            return None
        if text not in self._embed_cache:
            self._embed_cache[text] = np.array(self._mrt.embed_style(text))
        return self._embed_cache[text]

    def compute_embed(
        self,
        prompts: list[str],
        weights: list[float],
        focus_suffix: str,
        alpha: float,
    ) -> np.ndarray | None:
        focus_embed = self._embed(focus_suffix) if focus_suffix else None

        if not prompts:
            if focus_embed is None:
                return None
            norm = np.linalg.norm(focus_embed)
            return focus_embed / norm if norm > 1e-8 else focus_embed

        result: np.ndarray | None = None
        w_sum = 0.0
        for prompt, w in zip(prompts, weights):
            if w < 0.01:
                continue
            e = self._embed(prompt)
            if e is None:
                continue
            result = e * w if result is None else result + e * w
            w_sum += w

        if result is None:
            return None
        if w_sum > 0:
            result /= w_sum

        if focus_embed is not None and alpha > 0:
            result = result + alpha * focus_embed

        norm = np.linalg.norm(result)
        if norm > 1e-8:
            result /= norm
        return result

    def set_style(
        self,
        prompts: list[str],
        weights: list[float],
        focus_suffix: str = "",
        alpha: float = 0.0,
        transition_s: float = 0.0,
    ):
        new_embed = self.compute_embed(prompts, weights, focus_suffix, alpha)
        if new_embed is None:
            return

        if transition_s <= 0:
            with self._style_lock:
                self._current_embed = new_embed
            return

        with self._style_lock:
            old = self._current_embed

        if old is None:
            with self._style_lock:
                self._current_embed = new_embed
            return

        if self._ramp_thread and self._ramp_thread.is_alive():
            self._ramp_cancel = True
            self._ramp_thread.join(timeout=0.1)

        self._ramp_cancel = False
        self._ramp_thread = threading.Thread(
            target=self._ramp_loop,
            args=(old.copy(), new_embed.copy(), transition_s),
            daemon=True,
        )
        self._ramp_thread.start()

    def _ramp_loop(self, old: np.ndarray, new: np.ndarray, duration_s: float):
        steps    = max(1, int(duration_s * 20))
        interval = 1.0 / 20
        for i in range(steps + 1):
            if self._ramp_cancel:
                return
            t       = i / steps
            blended = (1 - t) * old + t * new
            norm    = np.linalg.norm(blended)
            if norm > 1e-8:
                blended /= norm
            with self._style_lock:
                self._current_embed = blended
            time.sleep(interval)

    # ── Mic + pitch detection ──────────────────────────────────────────────────

    def _mic_callback(self, indata: np.ndarray, frames: int, _time, _status):
        """Runs on the sounddevice audio thread every BLOCKSIZE samples (~40 ms)."""
        # Accumulate into the rolling pitch buffer
        mono = indata[:, 0]
        self._mic_level = float(np.abs(mono).mean())

        n = len(mono)
        self._pitch_buf = np.roll(self._pitch_buf, -n)
        self._pitch_buf[-n:] = mono

        # Detect pitch on the accumulated window
        freq, conf = detect_pitch(
            self._pitch_buf,
            SAMPLE_RATE,
            confidence_threshold=self.confidence_threshold,
        )
        midi = hz_to_midi(freq) if freq is not None else None

        self._detected_freq       = freq
        self._detected_midi       = midi
        self._detected_confidence = conf

        # Update RT2 notes conditioning via NoteStateTracker
        notes = self._note_tracker.update(midi)
        with self._notes_lock:
            self._midi_notes = notes if midi is not None else None

    def start_mic(self, device=None):
        if self._mic_stream is not None:
            return
        self._note_tracker.reset()
        self._midi_notes = None
        self._mic_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCKSIZE,
            device=device,
            callback=self._mic_callback,
        )
        self._mic_stream.start()
        self.mic_enabled = True
        # Store the resolved device name for status display.
        try:
            idx = self._mic_stream.device
            self._active_input_name = sd.query_devices(idx)["name"]
        except Exception:
            self._active_input_name = "unknown"

    def stop_mic(self):
        if self._mic_stream:
            self._mic_stream.stop()
            self._mic_stream.close()
            self._mic_stream = None
        self.mic_enabled = False
        self._active_input_name = ""
        self._mic_level = 0.0
        self._detected_freq = None
        self._detected_midi = None
        self._note_tracker.reset()
        with self._notes_lock:
            self._midi_notes = None

    # ── Playback ──────────────────────────────────────────────────────────────

    def _out_callback(self, outdata: np.ndarray, frames: int, _time, _status):
        chunk      = self._ring.read(frames)
        outdata[:] = chunk * self._volume

    def _open_output_stream(self) -> sd.OutputStream:
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCKSIZE,
            device=self.preferred_out_device,  # None = system default
            callback=self._out_callback,
        )
        stream.start()
        return stream

    def _device_monitor_loop(self):
        """
        Poll output stream health and default device name every second.

        Restarts the stream when either:
          - the stream became inactive or errored (e.g. macOS reconfigured the
            CoreAudio graph when a device was added/removed), or
          - the system default output device name changed.

        Opening the new stream before closing the old one keeps audio continuous.
        _out_device_name is only updated after a successful open so a transient
        failure (device not yet initialised) is automatically retried next poll.
        """
        while self._playing:
            time.sleep(1.0)
            if not self._playing:
                break

            stream_healthy = self._sd_out is not None and self._sd_out.active

            try:
                name = sd.query_devices(kind="output")["name"]
            except Exception:
                name = self._out_device_name  # query failed; keep current name

            device_changed = name != self._out_device_name

            if stream_healthy and not device_changed:
                continue

            old = self._sd_out
            try:
                new_stream = self._open_output_stream()
            except Exception as e:
                print(f"[engine] output '{name}' not ready, will retry: {e}")
                continue

            self._out_device_name = name
            self._sd_out = new_stream

            if old is not None:
                try:
                    old.stop()
                    old.close()
                except Exception:
                    pass

    def play(self) -> str:
        if not self._loaded:
            return "Load a model first."
        if self._playing:
            return "Already playing."
        self._playing = True

        try:
            self._out_device_name = sd.query_devices(kind="output")["name"]
        except Exception:
            self._out_device_name = ""

        self._sd_out = self._open_output_stream()

        self._device_monitor = threading.Thread(
            target=self._device_monitor_loop, daemon=True
        )
        self._device_monitor.start()

        self._mlx.submit(self._gen_loop)
        return "▶ Playing"

    def pause(self) -> str:
        self._playing = False
        if self._sd_out:
            self._sd_out.stop()
            self._sd_out.close()
            self._sd_out = None
        return "⏸ Paused"

    def set_volume(self, v: float):
        self._volume = float(np.clip(v, 0.0, 1.0))

    def _gen_loop(self):
        """Runs exclusively on the MLX thread — do not submit other MLX work while playing."""
        state = None
        while self._playing:
            if self._ring.buffered_seconds > 4.0:
                time.sleep(0.2)
                continue

            with self._style_lock:
                style = self._current_embed

            if style is None:
                time.sleep(0.05)
                continue

            with self._notes_lock:
                notes = self._midi_notes

            try:
                wav, state = self._mrt.generate(
                    style=style,
                    notes=notes,
                    frames=FRAMES_CHUNK,
                    state=state,
                )
                self._ring.write(np.array(wav.samples))
            except Exception as e:
                print(f"[gen] {e}")
                state = None
                time.sleep(0.5)

    # ── Status (read-only, for UI) ─────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def buffer_s(self) -> float:
        return self._ring.buffered_seconds

    @property
    def mic_level(self) -> float:
        return self._mic_level

    @property
    def detected_freq(self) -> float | None:
        return self._detected_freq

    @property
    def detected_midi(self) -> int | None:
        return self._detected_midi

    @property
    def detected_note_name(self) -> str:
        m = self._detected_midi
        return midi_to_name(m) if m is not None else "—"

    @property
    def detected_confidence(self) -> float:
        return self._detected_confidence

    @property
    def active_input_device(self) -> str:
        return self._active_input_name
