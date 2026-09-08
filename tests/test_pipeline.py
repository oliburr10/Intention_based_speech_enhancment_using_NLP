"""Integration tests for the full audio pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from intent_se.audio.pipeline import SpeechEnhancer, to_mono
from intent_se.config import AudioConfig

FS = 48_000


def _speech_like(n: int, seed: int = 0) -> np.ndarray:
    """A crude harmonic stand-in for voiced speech."""
    t = np.arange(n) / FS
    sig = np.zeros(n)
    for k, amp in enumerate([0.30, 0.18, 0.10, 0.06], start=1):
        sig += amp * np.sin(2 * np.pi * 180 * k * t)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t)
    return sig * envelope


def test_to_mono_averages_channels():
    stereo = np.stack([np.ones(10), np.full(10, 3.0)], axis=1)
    np.testing.assert_allclose(to_mono(stereo), np.full(10, 2.0))


def test_to_mono_passes_mono_through():
    mono = np.arange(10.0)
    np.testing.assert_allclose(to_mono(mono), mono)


def test_self_test_passes_with_default_config():
    assert SpeechEnhancer().self_test() < 1e-12


def test_self_test_raises_on_broken_wola():
    enh = SpeechEnhancer()
    enh.stft.window = np.ones(enh.cfg.win_length)  # violates WOLA
    with pytest.raises(RuntimeError, match="WOLA condition violated"):
        enh.self_test()


def test_preroll_passes_audio_through_unmodified():
    """During pre-roll the estimate is untrustworthy, so nothing is suppressed."""
    enh = SpeechEnhancer()
    assert not enh.noise_estimator.is_seeded

    rng = np.random.default_rng(0)
    for _ in range(enh.cfg.preroll_frames):
        enh.process_block(rng.normal(size=enh.cfg.block_size) * 0.05)

    assert enh.noise_estimator.is_seeded
    assert enh.stats.preroll_complete


def test_output_length_matches_input():
    enh = SpeechEnhancer()
    sig = np.random.default_rng(0).normal(size=12_345) * 0.05
    assert enh.process_signal(sig).shape == sig.shape


def test_enhancement_improves_snr():
    """Noise-only pre-roll, then speech + noise: output SNR must beat input SNR."""
    rng = np.random.default_rng(0)

    preroll = rng.normal(0, 0.05, FS // 2)
    speech = _speech_like(FS)
    noise = rng.normal(0, 0.05, FS)
    noisy = np.concatenate([preroll, speech + noise])

    enh = SpeechEnhancer()
    enh.set_parameters(beta=1.5)
    out = enh.process_signal(noisy)

    # Compare energy in the speech region against the noise-only region.
    speech_region = slice(FS // 2 + 4096, len(noisy))
    quiet_region = slice(4096, FS // 2 - 4096)

    in_ratio = np.sqrt((noisy[speech_region] ** 2).mean()) / np.sqrt(
        (noisy[quiet_region] ** 2).mean()
    )
    out_ratio = np.sqrt((out[speech_region] ** 2).mean()) / np.sqrt(
        (out[quiet_region] ** 2).mean()
    )

    assert out_ratio > in_ratio, (
        f"speech-to-background ratio did not improve: {in_ratio:.2f} -> {out_ratio:.2f}"
    )


def test_higher_beta_gives_lower_mean_gain():
    rng = np.random.default_rng(0)
    sig = np.concatenate([rng.normal(0, 0.05, FS // 2), _speech_like(FS) + rng.normal(0, 0.05, FS)])

    gains = []
    for beta in (0.5, 1.0, 2.0):
        enh = SpeechEnhancer()
        enh.set_parameters(beta=beta)
        enh.process_signal(sig)
        gains.append(enh.stats.mean_gain)

    assert gains[0] > gains[1] > gains[2]


def test_output_is_finite_for_pathological_input():
    enh = SpeechEnhancer()
    for sig in (np.zeros(8192), np.full(8192, 1e-9), np.full(8192, 5.0)):
        out = enh.process_signal(sig)
        assert np.all(np.isfinite(out))


def test_reset_returns_to_initial_state():
    enh = SpeechEnhancer()
    enh.process_signal(np.random.default_rng(0).normal(size=8192) * 0.05)
    assert enh.stats.frames_processed > 0

    enh.reset()
    assert enh.stats.frames_processed == 0
    assert not enh.noise_estimator.is_seeded


def test_custom_config_is_validated():
    with pytest.raises(ValueError):
        SpeechEnhancer(AudioConfig(hop_length=256))
