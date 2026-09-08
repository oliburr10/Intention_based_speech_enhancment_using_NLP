"""Parametric Wiener filter with an externally tunable suppression exponent.

The standard Wiener gain minimises the mean square error between the estimated
and true clean speech:

.. math::

    G(k) = \\frac{\\xi(k)}{1 + \\xi(k)}

where :math:`\\xi` is the a priori SNR. That is optimal under its assumptions
but has two practical problems:

1. High-frequency bins usually have lower SNR, so the filter suppresses them
   hard and starts behaving like a low-pass filter -- speech sounds dull.
2. The gain curve is fixed by the SNR estimate. There is no way to make the
   suppression gentler or more aggressive.

The parametric form introduces an exponent :math:`\\beta`:

.. math::

    G(k) = \\left( \\frac{\\xi(k)}{1 + \\xi(k)} \\right)^{\\beta}

Low :math:`\\beta` flattens the curve and preserves more high-frequency content;
high :math:`\\beta` steepens it and suppresses noise more aggressively.
:math:`\\beta = 1` recovers the standard Wiener filter exactly.

**This exponent is the parameter the NLP side controls.** Without a tunable
parameter there would be nothing for the classifier output to change; every
part of the NLP pipeline is ultimately working toward updating it.
"""

from __future__ import annotations

import numpy as np

from intent_se.config import WienerConfig

__all__ = ["ParametricWienerFilter"]

_EPS = 1e-12


class ParametricWienerFilter:
    """Spectral gain computation with a decision-directed a priori SNR.

    Parameters
    ----------
    n_bins:
        Number of frequency bins.
    config:
        Filter parameters. ``beta``, ``output_gain`` and ``tilt`` are the
        NLP-controllable ones and can be changed at any time via
        :meth:`set_parameters`.
    sample_rate, n_fft:
        Needed only to build the tilt shelf. Defaults match ``AudioConfig``.
    """

    def __init__(
        self,
        n_bins: int,
        config: WienerConfig | None = None,
        sample_rate: int = 48_000,
        n_fft: int = 2048,
    ) -> None:
        cfg = config or WienerConfig()
        self.n_bins = n_bins
        self.sample_rate = sample_rate
        self.n_fft = n_fft

        self.beta = float(cfg.beta)
        self.gain_floor = float(cfg.gain_floor)
        self.dd_alpha = float(cfg.dd_alpha)
        self.output_gain = float(cfg.output_gain)
        self.tilt = float(cfg.tilt)
        self._xi_min = 10.0 ** (cfg.snr_min_db / 10.0)

        # Previous frame's clean-speech power estimate, for the
        # decision-directed a priori SNR.
        self._prev_clean_power = np.zeros(n_bins)
        self._xi = np.full(n_bins, self._xi_min)

        # Normalised bin frequencies in [0, 1], used by the tilt shelf.
        self._norm_freq = np.linspace(0.0, 1.0, n_bins)

    # ------------------------------------------------------------------
    # Parameter control -- this is the NLP -> DSP entry point
    # ------------------------------------------------------------------

    def set_parameters(
        self,
        *,
        beta: float | None = None,
        gain_floor: float | None = None,
        output_gain: float | None = None,
        tilt: float | None = None,
    ) -> None:
        """Update filter parameters while the stream is running.

        All arguments are keyword-only and optional; omitted parameters keep
        their current value. Values are clamped to safe ranges so a bad
        classifier output cannot destabilise the audio path.

        Parameters
        ----------
        beta:
            Suppression exponent, clamped to ``[0.1, 4.0]``.
        gain_floor:
            Minimum per-bin gain, clamped to ``[0.0, 1.0]``.
        output_gain:
            Broadband linear gain, clamped to ``[0.1, 4.0]``.
        tilt:
            High-frequency tilt in dB, clamped to ``[-12.0, 12.0]``.
        """
        if beta is not None:
            self.beta = float(np.clip(beta, 0.1, 4.0))
        if gain_floor is not None:
            self.gain_floor = float(np.clip(gain_floor, 0.0, 1.0))
        if output_gain is not None:
            self.output_gain = float(np.clip(output_gain, 0.1, 4.0))
        if tilt is not None:
            self.tilt = float(np.clip(tilt, -12.0, 12.0))

    @property
    def parameters(self) -> dict[str, float]:
        """Current parameter values, as a plain dict for logging."""
        return {
            "beta": self.beta,
            "gain_floor": self.gain_floor,
            "output_gain": self.output_gain,
            "tilt": self.tilt,
        }

    def reset(self) -> None:
        """Clear the decision-directed state."""
        self._prev_clean_power[:] = 0.0
        self._xi[:] = self._xi_min

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    @property
    def a_priori_snr(self) -> np.ndarray:
        """The most recent a priori SNR estimate, one value per bin."""
        return self._xi.copy()

    def compute_gain(self, power: np.ndarray, noise_psd: np.ndarray) -> np.ndarray:
        """Compute the spectral gain for one frame.

        Uses the Ephraim-Malah decision-directed estimator for the a priori
        SNR, which blends the previous frame's clean-speech estimate with the
        current maximum-likelihood estimate. This is what keeps musical noise
        under control: a purely instantaneous SNR estimate fluctuates wildly
        between frames and the gain fluctuates with it.

        Parameters
        ----------
        power:
            Periodogram of the current frame, shape ``(n_bins,)``.
        noise_psd:
            Noise PSD estimate from IMCRA, shape ``(n_bins,)``.

        Returns
        -------
        np.ndarray
            Real gain in ``[gain_floor, 1]`` per bin, shape ``(n_bins,)``.
        """
        noise_psd = np.maximum(noise_psd, _EPS)

        # A posteriori SNR.
        gamma = power / noise_psd

        # Decision-directed a priori SNR.
        xi = self.dd_alpha * (self._prev_clean_power / noise_psd) + (
            1.0 - self.dd_alpha
        ) * np.maximum(gamma - 1.0, 0.0)
        xi = np.maximum(xi, self._xi_min)
        self._xi = xi

        # Parametric Wiener gain.
        gain = (xi / (1.0 + xi)) ** self.beta

        # Gain floor: never zero a bin out completely. A fully suppressed bin
        # that reappears in the next frame is exactly what musical noise is.
        gain = np.maximum(gain, self.gain_floor)

        if self.tilt != 0.0:
            # Linear-in-frequency shelf, 0 dB at DC to `tilt` dB at Nyquist.
            gain = gain * 10.0 ** (self.tilt * self._norm_freq / 20.0)

        gain = np.clip(gain * self.output_gain, 0.0, 4.0)

        # Feed the enhanced power forward for the next frame's DD estimate.
        self._prev_clean_power = (gain**2) * power

        return gain

    def __call__(self, spectrum: np.ndarray, noise_psd: np.ndarray) -> np.ndarray:
        """Apply the filter to a complex spectrum.

        Parameters
        ----------
        spectrum:
            Complex spectrum of the current frame.
        noise_psd:
            Noise PSD estimate from IMCRA.

        Returns
        -------
        np.ndarray
            Enhanced complex spectrum. Phase is untouched -- only the magnitude
            is modified, which is standard for this class of enhancement.
        """
        power = np.abs(spectrum) ** 2
        return spectrum * self.compute_gain(power, noise_psd)
