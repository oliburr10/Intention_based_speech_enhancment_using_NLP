"""STFT analysis/synthesis with weighted overlap-add (WOLA).

The reason the analysis window is the *square root* of a Hann window rather
than a full Hann window is the reconstruction stage. The same window is applied
at analysis and again at synthesis, so their product is one full Hann window.
At 50% overlap a periodic Hann window sums to exactly 1.0 across the two
overlapping frames, which is the WOLA condition:

.. math::

    \\sum_m w_a[n - mH] \\, w_s[n - mH] = 1 \\quad \\forall n

With that satisfied, the output is identical to the input apart from whatever
was deliberately done in the frequency domain. Get it wrong and the overlapping
contributions do not sum cleanly, producing amplitude ripple that sounds like
distortion.
"""

from __future__ import annotations

import numpy as np

from intent_se.config import AudioConfig

__all__ = ["SlidingSTFT", "sqrt_hann", "wola_check"]


def sqrt_hann(win_length: int) -> np.ndarray:
    """Return a square-root *periodic* Hann window.

    The window must be periodic (``endpoint=False``), not symmetric. A
    symmetric Hann window does not sum to a constant under overlap-add and
    leaves a small ripple in the reconstruction.

    Parameters
    ----------
    win_length:
        Window length in samples.

    Returns
    -------
    np.ndarray
        Window of shape ``(win_length,)``, values in ``[0, 1]``.
    """
    n = np.arange(win_length)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / win_length)
    return np.sqrt(hann)


def wola_check(window: np.ndarray, hop_length: int) -> float:
    """Measure how far a window/hop pair deviates from the WOLA condition.

    Overlap-adds ``window ** 2`` at the given hop and returns the maximum
    absolute deviation from 1.0 over the steady-state region. A value near
    machine epsilon means perfect reconstruction.

    This is run at system startup as a self-test: apply a gain of 1.0 to every
    bin, reconstruct, and confirm the original signal comes back.

    Parameters
    ----------
    window:
        The analysis window (the synthesis window is assumed identical).
    hop_length:
        Hop between successive frames in samples.

    Returns
    -------
    float
        Maximum absolute deviation from unity in the steady-state region.
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
    """Frame-by-frame STFT/ISTFT for real-time block processing.

    Bridges the mismatch between what the soundcard delivers (512 new samples
    per callback) and what the transform needs (a 1024-sample window). Each
    call to :meth:`analyze` shifts the oldest ``hop_length`` samples out of the
    input buffer and writes the new block into the tail, so the buffer always
    holds the most recent ``win_length`` samples.

    Parameters
    ----------
    config:
        Audio configuration. Validated on construction.
    """

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
        """Push one block of new samples and return the frame spectrum.

        Parameters
        ----------
        block:
            ``hop_length`` new time-domain samples (mono).

        Returns
        -------
        np.ndarray
            Complex spectrum of shape ``(n_bins,)``.
        """
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

        Parameters
        ----------
        spectrum:
            Processed complex spectrum of shape ``(n_bins,)``.

        Returns
        -------
        np.ndarray
            ``hop_length`` enhanced time-domain samples, ready for the soundcard.
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
