"""Tests for the STFT/WOLA analysis-synthesis path."""

from __future__ import annotations

import numpy as np
import pytest

from intent_se.audio.stft import SlidingSTFT, sqrt_hann, wola_check
from intent_se.config import AudioConfig


def test_sqrt_hann_is_periodic_not_symmetric():
    """A periodic window starts at 0 and does not return to 0 at the end."""
    w = sqrt_hann(1024)
    assert w[0] == pytest.approx(0.0, abs=1e-12)
    assert w[-1] > 0.0
    assert len(w) == 1024


def test_wola_condition_holds_at_50_percent_overlap():
    """sqrt-Hann squared must overlap-add to exactly 1.0."""
    deviation = wola_check(sqrt_hann(1024), 512)
    assert deviation < 1e-12


def test_wola_condition_fails_at_wrong_hop():
    """A hop that is not win/2 breaks the condition -- guards the assumption."""
    deviation = wola_check(sqrt_hann(1024), 300)
    assert deviation > 1e-3


def test_perfect_reconstruction_of_a_real_signal():
    """Analysis then synthesis with unit gain must return the input."""
    cfg = AudioConfig()
    stft = SlidingSTFT(cfg)

    rng = np.random.default_rng(0)
    n_blocks = 40
    signal = rng.normal(size=n_blocks * cfg.hop_length)

    out = np.zeros_like(signal)
    for i in range(n_blocks):
        sl = slice(i * cfg.hop_length, (i + 1) * cfg.hop_length)
        out[sl] = stft.synthesize(stft.analyze(signal[sl]))

    # The first frame is still filling the buffer, so compare from block 2 on.
    # Output lags the input by one window minus one hop.
    lag = cfg.win_length - cfg.hop_length
    start = 2 * cfg.hop_length
    np.testing.assert_allclose(
        out[start:], signal[start - lag : len(signal) - lag], atol=1e-10
    )


def test_analyze_rejects_wrong_block_size():
    stft = SlidingSTFT()
    with pytest.raises(ValueError, match="Expected a block"):
        stft.analyze(np.zeros(256))


def test_spectrum_has_expected_bin_count():
    cfg = AudioConfig()
    stft = SlidingSTFT(cfg)
    spectrum = stft.analyze(np.zeros(cfg.hop_length))
    assert spectrum.shape == (cfg.n_bins,)
    assert cfg.n_bins == 1025


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"hop_length": 256, "block_size": 256}, "50% overlap"),
        ({"n_fft": 512}, "must be >="),
        ({"block_size": 256}, "one hop per callback"),
    ],
)
def test_invalid_configurations_are_rejected(kwargs, message):
    cfg = AudioConfig(**kwargs)
    with pytest.raises(ValueError, match=message):
        cfg.validate()
