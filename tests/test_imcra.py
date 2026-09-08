"""Tests for the IMCRA noise estimator."""

from __future__ import annotations

import numpy as np
import pytest

from intent_se.audio.imcra import IMCRA
from intent_se.config import IMCRAConfig


def test_search_window_is_u_times_v():
    """D = U * V = 120 is the documented minimum-search window."""
    cfg = IMCRAConfig()
    assert cfg.d == 120
    assert cfg.u * cfg.v == cfg.d


def test_seeding_uses_the_per_bin_minimum():
    """A transient in the pre-roll must not inflate the noise floor."""
    est = IMCRA(n_bins=8)
    preroll = np.ones((5, 8))
    preroll[2] = 100.0  # a transient in one frame
    est.seed(preroll)
    np.testing.assert_allclose(est.noise_psd, np.ones(8))


def test_seed_rejects_wrong_bin_count():
    est = IMCRA(n_bins=8)
    with pytest.raises(ValueError, match="expected 8"):
        est.seed(np.ones((5, 16)))


def test_tracks_a_stationary_noise_floor():
    """Given stationary noise, the estimate should converge near its power."""
    rng = np.random.default_rng(0)
    n_bins = 129
    est = IMCRA(n_bins=n_bins)

    true_power = 0.25
    frames = rng.exponential(true_power, size=(200, n_bins))

    est.seed(frames[:5])
    for frame in frames[5:]:
        noise_psd = est.update(frame)

    # Within a factor of ~3 of the truth is the realistic bar for a minimum-
    # tracking estimator on exponentially distributed periodograms.
    assert 0.3 * true_power < noise_psd.mean() < 3.0 * true_power


def test_estimate_does_not_chase_speech_energy():
    """A sustained high-energy burst must not be absorbed as noise."""
    rng = np.random.default_rng(1)
    n_bins = 65
    est = IMCRA(n_bins=n_bins)

    noise = rng.exponential(0.1, size=(150, n_bins))
    est.seed(noise[:5])
    for frame in noise[5:]:
        est.update(frame)
    before = est.noise_psd.mean()

    # 60 frames of far louder "speech".
    for _ in range(60):
        est.update(rng.exponential(5.0, size=n_bins))
    after = est.noise_psd.mean()

    # The estimate rises somewhat but must stay far below the burst power.
    assert after < 1.0, f"noise estimate absorbed speech energy: {after:.3f}"
    assert after >= before


def test_update_before_seeding_degrades_gracefully():
    """Calling update() first must not divide by zero."""
    est = IMCRA(n_bins=16)
    assert not est.is_seeded
    out = est.update(np.full(16, 0.5))
    assert est.is_seeded
    assert np.all(np.isfinite(out))


def test_output_is_always_finite_and_positive():
    rng = np.random.default_rng(2)
    est = IMCRA(n_bins=33)
    est.seed(np.zeros((5, 33)))  # pathological: silent pre-roll
    for _ in range(50):
        out = est.update(rng.exponential(1.0, size=33))
        assert np.all(np.isfinite(out))
        assert np.all(out > 0)
