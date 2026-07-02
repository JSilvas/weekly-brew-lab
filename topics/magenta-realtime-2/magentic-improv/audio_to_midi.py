"""
Real-time audio-to-MIDI pitch detection using autocorrelation.

Detects the fundamental frequency from a mono audio buffer, converts to a
MIDI note number, and tracks onset vs. sustain state for RT2 conditioning.

No external dependencies beyond numpy.
"""

from __future__ import annotations

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_pitch(
    buffer: np.ndarray,
    sample_rate: int,
    fmin: float = 80.0,
    fmax: float = 1200.0,
    confidence_threshold: float = 0.4,
    silence_rms: float = 0.005,
) -> tuple[float | None, float]:
    """
    Detect fundamental frequency using FFT autocorrelation.

    Args:
        buffer: Mono float32 audio (any length ≥ sample_rate / fmin samples).
        sample_rate: Sample rate of the buffer.
        fmin: Minimum detectable frequency in Hz (lowest MIDI note to track).
        fmax: Maximum detectable frequency in Hz.
        confidence_threshold: Normalised autocorrelation peak required (0–1).
        silence_rms: RMS below this value is treated as silence.

    Returns:
        (freq_hz, confidence) — freq_hz is None when no pitch is detected or
        input is below the silence threshold.
    """
    buf = buffer.flatten()

    if np.sqrt(np.mean(buf ** 2)) < silence_rms:
        return None, 0.0

    buf = buf - buf.mean()

    # FFT-based autocorrelation (O(n log n) vs O(n²) for np.correlate)
    n = len(buf)
    fft = np.fft.rfft(buf, n=2 * n)
    corr = np.fft.irfft(fft * np.conj(fft))[:n]

    if corr[0] < 1e-10:
        return None, 0.0

    corr /= corr[0]

    lag_min = int(sample_rate / fmax)
    lag_max = min(int(sample_rate / fmin), n - 1)

    if lag_min >= lag_max:
        return None, 0.0

    region = corr[lag_min:lag_max]
    peak_local = int(np.argmax(region))
    peak_lag = peak_local + lag_min
    peak_val = corr[peak_lag]

    if peak_val < confidence_threshold:
        return None, float(peak_val)

    # Parabolic interpolation for sub-sample frequency accuracy
    if 0 < peak_lag < n - 1:
        alpha, beta, gamma = corr[peak_lag - 1], corr[peak_lag], corr[peak_lag + 1]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-10:
            peak_lag = peak_lag + 0.5 * (alpha - gamma) / denom

    return float(sample_rate / peak_lag), float(peak_val)


def hz_to_midi(freq: float) -> int | None:
    """Convert a frequency in Hz to the nearest MIDI note (0–127)."""
    if freq <= 0:
        return None
    note = round(69 + 12 * np.log2(freq / 440.0))
    return note if 0 <= note <= 127 else None


def midi_to_name(midi: int) -> str:
    """Return human-readable note name, e.g. 69 → 'A4'."""
    octave = (midi // 12) - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


class NoteStateTracker:
    """
    Tracks note onset vs. sustain for RT2 conditioning.

    RT2 notes array semantics:
        -1  masked  (ignore this pitch)
         0  off
         1  sustain (pitch was on last frame, still on)
         2  onset   (pitch just started)
         3  free    (model decides onset/sustain — used when unsure)
    """

    def __init__(self, n_pitches: int = 128):
        self._n = n_pitches
        self._active: int | None = None   # currently held MIDI note
        self._frames_held = 0

    def update(self, midi: int | None) -> list[int]:
        """
        Given the currently detected MIDI note (or None for silence), return
        a 128-element notes array suitable for RT2's `notes` parameter.
        """
        arr = [0] * self._n

        if midi is None:
            self._active = None
            self._frames_held = 0
            return arr

        if midi != self._active:
            arr[midi] = 2           # onset
            self._active = midi
            self._frames_held = 1
        else:
            arr[midi] = 1           # sustain
            self._frames_held += 1

        return arr

    def reset(self) -> None:
        self._active = None
        self._frames_held = 0

    @property
    def active_note(self) -> int | None:
        return self._active

    @property
    def frames_held(self) -> int:
        return self._frames_held
