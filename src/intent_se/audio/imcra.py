from __future__ import annotations
import numpy as np
from intent_se.config import IMCRAConfig
__all__ = ["IMCRA"]

_EPS = 1e-12


def _freq_smoothing_window(width: int = 5) -> np.ndarray:
    """Short Hann window used to smooth the periodogram across frequency."""
    w = np.hanning(width + 2)[1:-1]
    return w / w.sum()


class _MinimumTracker:
    """Two-stage minimum tracker over ``u * v`` frames.
    This is what makes the search window ``D = u * v = 120`` frames long while
    only requiring ``u + 1`` buffers rather than ``D`` of them.
    """

    def __init__(self, n_bins: int, u: int, v: int) -> None:
        self.u = u
        self.v = v
        self._store = np.full((u, n_bins), np.inf)
        self._running = np.full(n_bins, np.inf)
        self._frame = 0
        self._slot = 0

    def reset(self, value: np.ndarray | None = None) -> None:
        """Reset the tracker, optionally priming every slot with ``value``."""
        fill = np.inf if value is None else value
        self._store[:] = fill
        self._running = np.full_like(self._running, np.inf) if value is None else value.copy()
        self._frame = 0
        self._slot = 0

    def update(self, x: np.ndarray) -> np.ndarray:
        """Feed one frame and return the current tracked minimum per bin."""
        self._running = np.minimum(self._running, x)
        self._frame += 1

        if self._frame >= self.v:
            self._store[self._slot] = self._running
            self._slot = (self._slot + 1) % self.u
            self._running = x.copy()
            self._frame = 0

        return np.minimum(self._store.min(axis=0), self._running)


class IMCRA:
    """Recursive noise PSD estimator.
    """

    def __init__(self, n_bins: int, config: IMCRAConfig | None = None) -> None:
        self.cfg = config or IMCRAConfig()
        self.n_bins = n_bins
        self._b = _freq_smoothing_window()

        self._tracker = _MinimumTracker(n_bins, self.cfg.u, self.cfg.v)
        self._tracker_tilde = _MinimumTracker(n_bins, self.cfg.u, self.cfg.v)

        self._s = np.zeros(n_bins)
        self._s_tilde = np.zeros(n_bins)
        self._lambda_d = np.zeros(n_bins)
        self._seeded = False

  
    @property
    def is_seeded(self) -> bool:
        """Whether :meth:`seed` has been called."""
        return self._seeded

    @property
    def noise_psd(self) -> np.ndarray:
        """Current noise PSD estimate, one value per bin."""
        return self._lambda_d.copy()

    def seed(self, preroll_power: np.ndarray) -> None:
        """Initialise the internal state from pre-roll background noise - Fixed!!
        """
        preroll_power = np.atleast_2d(np.asarray(preroll_power, dtype=np.float64))
        if preroll_power.shape[1] != self.n_bins:
            raise ValueError(
                f"Pre-roll has {preroll_power.shape[1]} bins, expected {self.n_bins}."
            )

        floor = np.maximum(preroll_power.min(axis=0), _EPS)

        self._lambda_d = floor.copy()
        self._s = floor.copy()
        self._s_tilde = floor.copy()
        self._tracker.reset(floor)
        self._tracker_tilde.reset(floor)
        self._seeded = True

  
    # Per-frame update

    def update(self, power: np.ndarray, xi: np.ndarray | None = None) -> np.ndarray:
      
        """Process one frame and return the updated noise PSD.
        """
        cfg = self.cfg
        power = np.asarray(power, dtype=np.float64)

        if not self._seeded:
            # Degrade gracefully rather than dividing by zero: treat the first
            # frame as the noise floor. Callers should use seed() instead.
            self.seed(power[None, :])
            return self.noise_psd

        # First iteration: rough noise-only decision
        s_f = np.convolve(power, self._b, mode="same")
        self._s = cfg.alpha_s * self._s + (1.0 - cfg.alpha_s) * s_f
        s_min = self._tracker.update(self._s)

        denom = np.maximum(cfg.beta * s_min, _EPS)
        gamma_min = power / denom
        zeta = self._s / denom

        # I == 1 marks bins judged to contain noise only.
        indicator = ((gamma_min < cfg.gamma0) & (zeta < cfg.zeta0)).astype(np.float64)

        #Second iteration: smooth using noise-only bins
        num = np.convolve(indicator * power, self._b, mode="same")
        den = np.convolve(indicator, self._b, mode="same")
        s_f_tilde = np.where(den > _EPS, num / np.maximum(den, _EPS), self._s_tilde)

        self._s_tilde = cfg.alpha_s * self._s_tilde + (1.0 - cfg.alpha_s) * s_f_tilde
        s_min_tilde = self._tracker_tilde.update(self._s_tilde)

        denom_t = np.maximum(cfg.beta * s_min_tilde, _EPS)
        gamma_min_tilde = power / denom_t
        zeta_tilde = self._s_tilde / denom_t

        #A priori speech-absence probability q
        q = np.zeros(self.n_bins)
        low = (gamma_min_tilde <= 1.0) & (zeta_tilde < cfg.zeta0)
        mid = (gamma_min_tilde > 1.0) & (gamma_min_tilde < cfg.gamma1) & (zeta_tilde < cfg.zeta0)
        q[low] = 1.0
        q[mid] = (cfg.gamma1 - gamma_min_tilde[mid]) / (cfg.gamma1 - 1.0)
        q = np.clip(q, 0.0, 1.0 - 1e-6)

        #Speech-presence probability p
        gamma = power / np.maximum(self._lambda_d, _EPS)
        if xi is None:
            xi = np.maximum(gamma - 1.0, _EPS)
        nu = gamma * xi / (1.0 + xi)
        ratio = (q / (1.0 - q)) * (1.0 + xi) * np.exp(-np.clip(nu, 0.0, 500.0))
        p = 1.0 / (1.0 + ratio)

        #Recursive averaging with speech-controlled adaptation
        # alpha -> 1 where speech is likely, so the estimate freezes rather
        # than absorbing speech energy.
        alpha_tilde = cfg.alpha_d + (1.0 - cfg.alpha_d) * p
        self._lambda_d = alpha_tilde * self._lambda_d + (1.0 - alpha_tilde) * power

        # Bias compensation: the minimum of a smoothed periodogram is a biased
        # estimate of the mean, so scale it back up.
        return np.maximum(cfg.beta * self._lambda_d, _EPS)
