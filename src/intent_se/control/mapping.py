"""Mapping (intent, severity) onto DSP parameter changes.

.. warning::
   **This module is the open research question of the thesis, and its numbers
   are not calibrated.**

   The *direction* of every adjustment below is theoretically grounded and was
   confirmed to work: manually changing ``beta`` during real-time operation
   produces clearly audible differences in noise suppression, exactly as the
   signal processing theory predicts. The wiring is correct.

   The *magnitude* is not derivable. If the severity scorer outputs 0.6 for one
   complaint and 0.8 for another, how much larger should the second parameter
   change be? Should the relationship be linear? Should it saturate? There is
   no analytical answer, because the right answer depends on how a real user
   perceives the difference between two adjustment levels in their own
   acoustic environment. Establishing it requires a structured listening study:
   recruit hearing-aid users, present controlled variations of each parameter
   across the severity range, have them rate which adjustments felt
   appropriate, and fit a mapping function to the results.

   The ``max_delta`` values in :data:`INTENT_ACTIONS` are therefore **plausible
   placeholders chosen to produce an audible but not destructive change**, not
   validated constants. Treat every number here as provisional until the
   perceptual study is run.

Direction rationale
-------------------

========================  ==================================================
Intent                    Response
========================  ==================================================
``TOO_NOISY``             Raise ``beta``: steepen the Wiener gain curve so
                          noise-dominated bins are suppressed harder. Also
                          lower ``gain_floor`` to allow deeper attenuation.
``SPEECH_UNCLEAR``        Lower ``beta``: flatten the curve so more
                          high-frequency detail survives, since over-
                          suppression is what makes speech sound muffled.
``TOO_LOUD``              Reduce broadband ``output_gain``.
``TOO_QUIET``             Raise broadband ``output_gain``.
``TOO_SHARP``             Apply negative ``tilt``: attenuate the high end,
                          where harshness and sibilance live, without
                          touching broadband level.
``BALANCED``              No complaint. Decay the parameters back toward
                          their defaults rather than holding whatever the
                          last complaint set.
========================  ==================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

from intent_se.config import NEUTRAL_CLASS, WienerConfig

__all__ = ["INTENT_ACTIONS", "IntentAction", "ParameterController", "ParameterUpdate"]


@dataclass(frozen=True)
class IntentAction:
    """Which parameter an intent moves, in which direction, and how far.

    Attributes
    ----------
    parameter:
        Name of the DSP parameter to change.
    direction:
        ``+1`` to increase, ``-1`` to decrease. **This part is grounded.**
    max_delta:
        Change applied at severity 1.0. **This part is an uncalibrated
        placeholder** -- see the module docstring.
    secondary:
        Optional additional ``{parameter: (direction, max_delta)}`` moves.
    """

    parameter: str
    direction: int
    max_delta: float
    secondary: dict[str, tuple[int, float]] = field(default_factory=dict)


#: The mapping table. Directions are grounded in signal processing theory;
#: magnitudes await the perceptual calibration study.
INTENT_ACTIONS: dict[str, IntentAction] = {
    "TOO_NOISY": IntentAction(
        parameter="beta",
        direction=+1,
        max_delta=1.0,
        secondary={"gain_floor": (-1, 0.04)},
    ),
    "SPEECH_UNCLEAR": IntentAction(
        parameter="beta",
        direction=-1,
        max_delta=0.5,
        secondary={"tilt": (+1, 3.0)},
    ),
    "TOO_LOUD": IntentAction(parameter="output_gain", direction=-1, max_delta=0.5),
    "TOO_QUIET": IntentAction(parameter="output_gain", direction=+1, max_delta=1.0),
    "TOO_SHARP": IntentAction(parameter="tilt", direction=-1, max_delta=8.0),
}


@dataclass
class ParameterUpdate:
    """One controller decision, with enough context to be logged and audited."""

    intent: str
    severity: float
    confidence: float
    applied: bool
    parameters: dict[str, float]
    reason: str = ""

    def __str__(self) -> str:
        status = "applied" if self.applied else "skipped"
        params = ", ".join(f"{k}={v:.3f}" for k, v in sorted(self.parameters.items()))
        return (
            f"[{status}] {self.intent} (severity {self.severity:.2f}, "
            f"confidence {self.confidence:.2f}) -> {params}"
            + (f"  # {self.reason}" if self.reason else "")
        )


class ParameterController:
    """Translates classifier and severity output into DSP parameter updates.

    Adjustments are **relative and cumulative**: each complaint nudges the
    current parameter values rather than jumping to an absolute target, so a
    user can say "still too noisy" repeatedly and keep moving in that
    direction. Values are clamped by the filter itself.

    Parameters
    ----------
    defaults:
        Baseline parameter values, restored by :meth:`reset` and decayed toward
        on ``BALANCED``.
    confidence_threshold:
        Classifications below this confidence are ignored. Acting on a guess is
        worse than doing nothing -- the user will simply rephrase.
    min_severity:
        Complaints milder than this are ignored, so trivial remarks do not
        cause parameter drift.
    neutral_decay:
        Fraction of the distance back to the defaults travelled on each
        ``BALANCED`` utterance.

    Examples
    --------
    >>> ctrl = ParameterController()
    >>> update = ctrl.apply("TOO_NOISY", severity=0.8, confidence=0.95)
    >>> update.applied
    True
    >>> update.parameters["beta"] > 1.0
    True
    """

    def __init__(
        self,
        defaults: WienerConfig | None = None,
        confidence_threshold: float = 0.35,
        min_severity: float = 0.05,
        neutral_decay: float = 0.5,
    ) -> None:
        cfg = defaults or WienerConfig()
        self.defaults = {
            "beta": cfg.beta,
            "gain_floor": cfg.gain_floor,
            "output_gain": cfg.output_gain,
            "tilt": cfg.tilt,
        }
        self.confidence_threshold = confidence_threshold
        self.min_severity = min_severity
        self.neutral_decay = neutral_decay

        self.parameters = dict(self.defaults)
        self.history: list[ParameterUpdate] = []

    # ------------------------------------------------------------------

    def reset(self) -> dict[str, float]:
        """Restore the default parameters and clear the history."""
        self.parameters = dict(self.defaults)
        self.history.clear()
        return dict(self.parameters)

    def apply(self, intent: str, severity: float, confidence: float = 1.0) -> ParameterUpdate:
        """Compute and record the parameter change for one classified sentence.

        Parameters
        ----------
        intent:
            Predicted class name.
        severity:
            Predicted severity in ``[0, 1]``.
        confidence:
            Classifier confidence in ``[0, 1]``.

        Returns
        -------
        ParameterUpdate
            The decision, including whether it was applied and why not.
        """
        severity = float(min(max(severity, 0.0), 1.0))

        def record(applied: bool, reason: str = "") -> ParameterUpdate:
            update = ParameterUpdate(
                intent=intent,
                severity=severity,
                confidence=confidence,
                applied=applied,
                parameters=dict(self.parameters),
                reason=reason,
            )
            self.history.append(update)
            return update

        if confidence < self.confidence_threshold:
            return record(False, f"confidence below {self.confidence_threshold}")

        if intent == NEUTRAL_CLASS:
            # The user is satisfied. Relax toward the defaults instead of
            # holding whatever the previous complaint set.
            for key, default in self.defaults.items():
                self.parameters[key] += (default - self.parameters[key]) * self.neutral_decay
            return record(True, "neutral -- decaying toward defaults")

        action = INTENT_ACTIONS.get(intent)
        if action is None:
            return record(False, f"no mapping defined for intent '{intent}'")

        if severity < self.min_severity:
            return record(False, f"severity below {self.min_severity}")

        # Linear severity -> magnitude. Whether the true relationship is linear
        # or saturating is precisely what the perceptual study must determine.
        self._nudge(action.parameter, action.direction, action.max_delta, severity)
        for param, (direction, max_delta) in action.secondary.items():
            self._nudge(param, direction, max_delta, severity)

        return record(True)

    def _nudge(self, parameter: str, direction: int, max_delta: float, severity: float) -> None:
        """Move one parameter by ``direction * max_delta * severity``."""
        self.parameters[parameter] = self.parameters[parameter] + direction * max_delta * severity

    # ------------------------------------------------------------------

    def bind(self, enhancer) -> None:  # noqa: ANN001 - avoids a circular import
        """Push the current parameters into a :class:`~intent_se.audio.SpeechEnhancer`.

        The enhancer clamps whatever it receives, so a wild classifier output
        cannot destabilise the audio path.
        """
        enhancer.set_parameters(**self.parameters)
