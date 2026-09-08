"""Tests for the NLP-to-DSP parameter controller."""

from __future__ import annotations

import pytest

from intent_se.control.mapping import INTENT_ACTIONS, ParameterController


def test_defaults_are_the_wiener_defaults():
    ctrl = ParameterController()
    assert ctrl.parameters == {
        "beta": 1.0,
        "gain_floor": 0.05,
        "output_gain": 1.0,
        "tilt": 0.0,
    }


def test_every_complaint_class_has_a_mapping():
    from intent_se.config import CLASS_ORDER, NEUTRAL_CLASS

    complaints = set(CLASS_ORDER) - {NEUTRAL_CLASS}
    assert complaints == set(INTENT_ACTIONS)


@pytest.mark.parametrize(
    "intent, parameter, expect_increase",
    [
        ("TOO_NOISY", "beta", True),
        ("SPEECH_UNCLEAR", "beta", False),
        ("TOO_LOUD", "output_gain", False),
        ("TOO_QUIET", "output_gain", True),
        ("TOO_SHARP", "tilt", False),
    ],
)
def test_adjustment_directions(intent, parameter, expect_increase):
    """The directions are the grounded part of the mapping -- pin them down."""
    ctrl = ParameterController()
    before = ctrl.parameters[parameter]
    ctrl.apply(intent, severity=0.8, confidence=0.9)
    after = ctrl.parameters[parameter]

    if expect_increase:
        assert after > before
    else:
        assert after < before


def test_magnitude_scales_with_severity():
    mild = ParameterController()
    mild.apply("TOO_NOISY", severity=0.2, confidence=0.9)

    severe = ParameterController()
    severe.apply("TOO_NOISY", severity=0.9, confidence=0.9)

    assert severe.parameters["beta"] > mild.parameters["beta"]


def test_adjustments_are_cumulative():
    """Repeating a complaint keeps moving in the same direction."""
    ctrl = ParameterController()
    ctrl.apply("TOO_NOISY", severity=0.5, confidence=0.9)
    first = ctrl.parameters["beta"]
    ctrl.apply("TOO_NOISY", severity=0.5, confidence=0.9)
    assert ctrl.parameters["beta"] > first


def test_low_confidence_is_ignored():
    ctrl = ParameterController(confidence_threshold=0.5)
    update = ctrl.apply("TOO_NOISY", severity=0.9, confidence=0.1)

    assert not update.applied
    assert "confidence" in update.reason
    assert ctrl.parameters["beta"] == 1.0


def test_trivial_severity_is_ignored():
    ctrl = ParameterController(min_severity=0.1)
    update = ctrl.apply("TOO_NOISY", severity=0.01, confidence=0.9)

    assert not update.applied
    assert ctrl.parameters["beta"] == 1.0


def test_balanced_decays_toward_defaults():
    ctrl = ParameterController(neutral_decay=0.5)
    ctrl.apply("TOO_NOISY", severity=1.0, confidence=0.9)
    raised = ctrl.parameters["beta"]
    assert raised > 1.0

    ctrl.apply("BALANCED", severity=0.0, confidence=0.9)
    assert 1.0 < ctrl.parameters["beta"] < raised


def test_repeated_balanced_converges_to_defaults():
    ctrl = ParameterController(neutral_decay=0.5)
    ctrl.apply("TOO_NOISY", severity=1.0, confidence=0.9)
    for _ in range(40):
        ctrl.apply("BALANCED", severity=0.0, confidence=0.9)
    assert ctrl.parameters["beta"] == pytest.approx(1.0, abs=1e-6)


def test_unknown_intent_is_rejected():
    ctrl = ParameterController()
    update = ctrl.apply("NOT_A_CLASS", severity=0.5, confidence=0.9)
    assert not update.applied
    assert "no mapping" in update.reason


def test_reset_restores_defaults_and_clears_history():
    ctrl = ParameterController()
    ctrl.apply("TOO_NOISY", severity=0.9, confidence=0.9)
    ctrl.reset()
    assert ctrl.parameters["beta"] == 1.0
    assert ctrl.history == []


def test_history_records_every_decision():
    ctrl = ParameterController()
    ctrl.apply("TOO_NOISY", severity=0.5, confidence=0.9)
    ctrl.apply("TOO_NOISY", severity=0.5, confidence=0.0)  # skipped
    assert len(ctrl.history) == 2
    assert [u.applied for u in ctrl.history] == [True, False]


def test_bind_pushes_parameters_into_the_enhancer():
    from intent_se.audio.pipeline import SpeechEnhancer

    enh = SpeechEnhancer()
    ctrl = ParameterController()
    ctrl.apply("TOO_NOISY", severity=0.8, confidence=0.9)
    ctrl.bind(enh)

    assert enh.parameters["beta"] == pytest.approx(ctrl.parameters["beta"])


def test_bound_parameters_are_clamped_by_the_filter():
    """Extreme cumulative adjustments must not escape the safe range."""
    from intent_se.audio.pipeline import SpeechEnhancer

    enh = SpeechEnhancer()
    ctrl = ParameterController()
    for _ in range(50):
        ctrl.apply("TOO_NOISY", severity=1.0, confidence=0.9)
    ctrl.bind(enh)

    assert 0.1 <= enh.parameters["beta"] <= 4.0
