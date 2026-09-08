from __future__ import annotations
import numpy as np
from intent_se.config import AudioConfig

__all__ = ["SlidingSTFT", "sqrt_hann", "wola_check"]


def sqrt_hann(win_length: int) -> np.ndarray:
        n = np.arange(win_length)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / win_length)
    return np.sqrt(hann)


def wola_check(window: np.ndarray, hop_length: int) -> float:
    """Measure how far a window/hop pair deviates from the WOLA condition.
    """
    win_length = len(window)
    n_frames = 8
    total = (n_frames - 1) * hop_length + win_length
    acc = np.zeros(total)
    for m in range(n_frames):
        acc[m * hop_length : m * hop_length + win_length] += window**2

    # Only the region covered by a full set of overlapping frames is steady.
    start = win_length - hop_length
    stop = total - (win_length - hop_length)
    return float(np.max(np.abs(acc[start:stop] - 1.0)))


class SlidingSTFT:
    def __init__(self, config: AudioConfig | None = None) -> None:
        self.cfg = config or AudioConfig()
        self.cfg.validate()

        self.window = sqrt_hann(self.cfg.win_length)

        # Input buffer holds the most recent ``win_length`` samples.
        self._in_buffer = np.zeros(self.cfg.win_length, dtype=np.float64)
        # Output buffer accumulates overlap-added frames.
        self._out_buffer = np.zeros(self.cfg.win_length, dtype=np.float64)

    @property
    def reconstruction_error(self) -> float:
        """Max deviation from the WOLA condition for this window/hop pair."""
        return wola_check(self.window, self.cfg.hop_length)

    def reset(self) -> None:
        """Clear both buffers. Call between independent streams."""
        self._in_buffer[:] = 0.0
        self._out_buffer[:] = 0.0

    def analyze(self, block: np.ndarray) -> np.ndarray:
        hop = self.cfg.hop_length
        if block.shape[0] != hop:
            raise ValueError(f"Expected a block of {hop} samples, got {block.shape[0]}.")

        # Slide: discard the oldest hop samples, append the new ones.
        self._in_buffer[:-hop] = self._in_buffer[hop:]
        self._in_buffer[-hop:] = block

        windowed = self._in_buffer * self.window
        # rfft zero-pads from win_length to n_fft automatically.
        return np.fft.rfft(windowed, n=self.cfg.n_fft)

    def synthesize(self, spectrum: np.ndarray) -> np.ndarray:
        """Invert one frame and overlap-add it, returning the next output block.
        """
        hop = self.cfg.hop_length

        frame = np.fft.irfft(spectrum, n=self.cfg.n_fft)
        # Discard the zero-padded tail; only the original window is meaningful.
        frame = frame[: self.cfg.win_length] * self.window

        self._out_buffer += frame
        out = self._out_buffer[:hop].copy()

        # Shift the accumulator forward and zero the newly exposed tail.
        self._out_buffer[:-hop] = self._out_buffer[hop:]
        self._out_buffer[-hop:] = 0.0

        return out
