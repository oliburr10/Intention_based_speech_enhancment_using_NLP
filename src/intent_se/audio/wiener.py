from __future__ import annotations
import numpy as np
from intent_se.config import WienerConfig

__all__ = ["ParametricWienerFilter"]

_EPS = 1e-12


class ParametricWienerFilter:
    """Spectral gain computation with a decision-directed a priori SNR.
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

    
    # Parameter control (this is the NLP -> DSP entry point)
    def set_parameters(
        self,
        *,
        beta: float | None = None,
        gain_floor: float | None = None,
        output_gain: float | None = None,
        tilt: float | None = None,
    ) -> None:
        """Update filter parameters while the stream is running
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


    # Per-frame processing
    @property
    def a_priori_snr(self) -> np.ndarray:
        """The most recent a priori SNR estimate, one value per bin."""
        return self._xi.copy()

    def compute_gain(self, power: np.ndarray, noise_psd: np.ndarray) -> np.ndarray:
        """Compute the spectral gain for one frame.
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
        power = np.abs(spectrum) ** 2
        return spectrum * self.compute_gain(power, noise_psd)
