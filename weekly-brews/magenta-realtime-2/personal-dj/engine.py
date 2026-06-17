"""
Magenta RT2 inference engine for Personal DJ.

MLX GPU streams are thread-local. Both model load and generate() MUST run on
the same thread. We use a single-worker ThreadPoolExecutor as a dedicated MLX
thread; all submissions to it run sequentially on that one thread.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

MAGENTA_HOME = os.environ.get(
    "MAGENTA_HOME",
    str(__import__("pathlib").Path.home() / "Documents" / "Magenta" / "magenta-rt-v2"),
)

SAMPLE_RATE    = 48_000
CHANNELS       = 2
FRAMES_CHUNK   = 50        # 2 s of audio per generate() call
BLOCKSIZE      = 1_920     # one codec frame = 40 ms
RING_MAXCHUNKS = 8         # ~16 s ceiling


class AudioRingBuffer:
    def __init__(self):
        self._q: deque[np.ndarray] = deque()
        self._partial: np.ndarray | None = None
        self._offset  = 0
        self._lock    = threading.Lock()

    def write(self, data: np.ndarray):
        with self._lock:
            if len(self._q) < RING_MAXCHUNKS:
                self._q.append(data.astype(np.float32))

    def read(self, n_frames: int) -> np.ndarray:
        out    = np.zeros((n_frames, CHANNELS), dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < n_frames:
                if self._partial is not None:
                    avail = len(self._partial) - self._offset
                    take  = min(avail, n_frames - filled)
                    out[filled:filled + take] = self._partial[self._offset:self._offset + take]
                    filled        += take
                    self._offset  += take
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


class DJEngine:
    def __init__(self):
        # Single-worker executor: all MLX ops (load + generate) run here
        self._mlx = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._mrt             = None
        self._mrt_state       = None
        self._style_lock      = threading.Lock()
        self._current_embed   = None
        self._embed_cache: dict[str, np.ndarray] = {}
        self._ring            = AudioRingBuffer()
        self._playing         = False
        self._volume          = 0.7
        self._loaded          = False
        self._sd_stream       = None
        self._ramp_thread: threading.Thread | None = None
        self._ramp_cancel     = False

    # ── Load (submitted to MLX thread) ───────────────────────────────────────

    def load(self, model_size: str = "mrt2_small") -> str:
        try:
            return self._mlx.submit(self._do_load, model_size).result(timeout=120)
        except Exception as e:
            return f"✗ Load failed: {e}"

    def _do_load(self, model_size: str) -> str:
        from magenta_rt import paths, MagentaRT2Mlxfn
        paths.set_magenta_home(MAGENTA_HOME)
        self._mrt = MagentaRT2Mlxfn(size=model_size)
        self._mrt_state = None
        self._loaded    = True
        self._embed_cache.clear()
        return f"✓ {model_size} loaded"

    # ── Embedding (embed_style uses TFLite/CPU — safe on any thread) ─────────

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

        if focus_suffix and alpha > 0:
            fe = self._embed(focus_suffix)
            if fe is not None:
                result = result + alpha * fe

        norm = np.linalg.norm(result)
        if norm > 1e-8:
            result /= norm
        return result

    def set_style(
        self,
        prompts: list[str],
        weights: list[float],
        focus_suffix: str,
        alpha: float,
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

        self._ramp_cancel  = False
        self._ramp_thread  = threading.Thread(
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

    # ── Playback ──────────────────────────────────────────────────────────────

    def play(self) -> str:
        if not self._loaded:
            return "Load a model first."
        if self._playing:
            return "Already playing."
        self._playing   = True
        self._mrt_state = None

        import sounddevice as sd

        def _cb(outdata, frames, _t, _s):
            chunk      = self._ring.read(frames)
            outdata[:] = chunk * self._volume

        self._sd_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCKSIZE,
            callback=_cb,
        )
        self._sd_stream.start()

        # Submit gen loop to the MLX thread (same thread the model was loaded on)
        self._mlx.submit(self._gen_loop)
        return "▶ Playing"

    def pause(self) -> str:
        self._playing = False
        if self._sd_stream:
            self._sd_stream.stop()
            self._sd_stream = None
        return "⏸ Paused"

    def set_volume(self, v: float):
        self._volume = float(np.clip(v, 0.0, 1.0))

    def _gen_loop(self):
        """Runs on the MLX thread — MUST stay on this thread for GPU streams."""
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

            try:
                wav, state = self._mrt.generate(
                    style=style,
                    notes=None,
                    frames=FRAMES_CHUNK,
                    state=state,
                )
                self._ring.write(np.array(wav.samples))
            except Exception as e:
                print(f"[engine] {e}")
                state = None
                time.sleep(0.5)

    # ── Status ────────────────────────────────────────────────────────────────

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def buffer_s(self) -> float:
        return self._ring.buffered_seconds
