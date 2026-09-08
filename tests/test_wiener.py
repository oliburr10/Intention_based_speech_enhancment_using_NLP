"""Tests for the parametric Wiener filter."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from intent_se.audio.wiener import ParametricWienerFilter
from intent_se.config import WienerConfig


def test_beta_one_recovers_the_standard_wiener_gain():
    """With beta = 1 the gain must equal xi / (1 + xi)."""
    filt = ParametricWienerFilter(n_bins=4, config=WienerConfig(beta=1.0, gain_floor=0.0,
                                                               dd_alpha=0.0))
    noise = np.ones(4)
    power = np.array([2.0, 5.0, 10.0, 100.0])

    gain = filt.compute_gain(power, noise)
    xi = np.maximum(power / noise - 1.0, 0.0)
    np.testing.assert_allclose(gain, xi / (1.0 + xi), rtol=1e-6)


def test_higher_beta_suppresses_more():
    """Steeper gain curve => smaller gain wherever gain < 1."""
    noise = np.ones(16)
    power = np.linspace(1.5, 20.0, 16)

    gains = []
    for beta in (0.5, 1.0, 2.0, 3.0):
        filt = ParametricWienerFilter(
            n_bins=16, config=WienerConfig(beta=beta, gain_floor=0.0, dd_alpha=0.0)
        )
        gains.append(filt.compute_gain(power, noise))

    for lower, higher in pairwise(gains):
        assert np.all(higher <= lower + 1e-12)


def test_gain_floor_is_respected():
    """No bin may be attenuated below the floor -- this prevents musical noise."""
    filt = ParametricWienerFilter(
        n_bins=8, config=WienerConfig(beta=3.0, gain_floor=0.05, dd_alpha=0.0)
    )
    # Power far below the noise floor => gain would otherwise go to ~0.
    gain = filt.compute_gain(np.full(8, 1e-6), np.ones(8))
    assert np.all(gain >= 0.05 - 1e-12)


def test_parameters_are_clamped_to_safe_ranges():
    """A wild classifier output must not destabilise the audio path."""
    filt = ParametricWienerFilter(n_bins=8)

    filt.set_parameters(beta=1e6, output_gain=-50.0, tilt=999.0, gain_floor=17.0)
    assert filt.beta == 4.0
    assert filt.output_gain == 0.1
    assert filt.tilt == 12.0
    assert filt.gain_floor == 1.0


def test_set_parameters_leaves_omitted_values_alone():
    filt = ParametricWienerFilter(n_bins=8)
    original = filt.parameters
    filt.set_parameters(beta=2.0)
    assert filt.beta == 2.0
    assert filt.output_gain == original["output_gain"]
    assert filt.tilt == original["tilt"]


def test_negative_tilt_attenuates_high_frequencies():
    """TOO_SHARP applies negative tilt; the high end must drop relative to DC."""
    filt = ParametricWienerFilter(
        n_bins=64, config=WienerConfig(beta=1.0, gain_floor=0.0, dd_alpha=0.0, tilt=-12.0)
    )
    gain = filt.compute_gain(np.full(64, 100.0), np.ones(64))
    assert gain[-1] < gain[0]


def test_applying_the_filter_preserves_phase():
    """Only the magnitude is modified."""
    rng = np.random.default_rng(0)
    spectrum = rng.normal(size=32) + 1j * rng.normal(size=32)
    filt = ParametricWienerFilter(n_bins=32)

    out = filt(spectrum, np.full(32, 0.1))
    # Gains are real and non-negative, so phase is unchanged.
    np.testing.assert_allclose(np.angle(out), np.angle(spectrum), atol=1e-9)


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_gain_never_exceeds_unity_without_extra_gain(beta):
    filt = ParametricWienerFilter(
        n_bins=32, config=WienerConfig(beta=beta, dd_alpha=0.0)
    )
    gain = filt.compute_gain(np.linspace(0.1, 500.0, 32), np.ones(32))
    assert np.all(gain <= 1.0 + 1e-9)
