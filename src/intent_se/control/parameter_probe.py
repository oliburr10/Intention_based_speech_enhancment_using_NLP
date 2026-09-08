"""Exploratory probe: do the DSP parameters respond when the NLP output moves them?

.. warning::
   **This module is not a finished controller. It is a test harness for an
   open problem, and its magnitudes are not validated.**

From the thesis:

    The major problem with the process of translating a user's complaint into a
    change of a parameter's value is that there is no mathematically accurate
    ground truth that relates a particular complaint to the corresponding value
    of that parameter. The general direction of each parameter manipulation can
    be understood through signal processing theory; for example, manipulating
    the suppression exponent β upwards in case the user complains about the
    noise will make the Wiener gain function grow steeper and hence suppress
    more strongly frequency bins that are affected by noise. However, since it
    is not possible to calculate analytically how strong the effect should be,
    there is an inevitable level of subjectivity in the relation between the
    numeric severity score of 0.6 and 0.8 and the actual change of the
    parameter. For example, one user can be satisfied with either of those
    levels while the other feels a great difference between them.

    Due to time constraints, a complete perceptual validation and mapping of
    the parameters was not implemented. This remains as an open challenge for
    further work.

What this module therefore *is*
-------------------------------

A way to exercise one parameter at a time and confirm that moving it produces
the effect signal processing theory predicts — that the wiring from a classified
sentence through to the audio path is intact and audible. That much was verified:
adjusting ``beta`` during real-time operation produces clearly audible changes in
noise suppression.

What it is *not*
----------------

A calibrated mapping from severity to magnitude. The ``probe_delta`` values in
:data:`INTENT_ACTIONS` are **arbitrary step sizes chosen to be audible but not
destructive**, so that the effect of a change can be heard at all. They carry no
perceptual meaning. Establishing real values requires a structured listening
study: recruiting hearing-aid users, presenting controlled variations of each
parameter across the severity range, having them rate which adjustments felt
appropriate, and fitting a mapping function to the results.

Directions, which *are* grounded
--------------------------------

Only the direction column below follows from signal processing theory. The step
sizes do not.

========================  ==================================================
Intent                    Direction of manipulation
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

__all__ = [
    "INTENT_ACTIONS",
    "PARAM_LIMITS",
    "IntentAction",
    "ParameterProbe",
    "ParameterUpdate",
]


@dataclass(frozen=True)
class IntentAction:
    """Which parameter an intent moves, in which direction, and by how much.

    Attributes
    ----------
    parameter:
        Name of the DSP parameter to change.
    direction:
        ``+1`` to increase, ``-1`` to decrease. **Grounded in signal processing
        theory.**
    probe_delta:
        Step applied at severity 1.0. **An arbitrary probe value, not a
        calibrated one** -- see the module docstring.
    secondary:
        Optional additional ``{parameter: (direction, probe_delta)}`` moves.
    """

    parameter: str
    direction: int
    probe_delta: float
    secondary: dict[str, tuple[int, float]] = field(default_factory=dict)


#: The probe table. Directions are grounded in signal processing theory; the
#: step sizes are arbitrary and await the perceptual calibration study.
INTENT_ACTIONS: dict[str, IntentAction] = {
    "TOO_NOISY": IntentAction(
        parameter="beta",
        direction=+1,
        probe_delta=1.0,
        secondary={"gain_floor": (-1, 0.04)},
    ),
    "SPEECH_UNCLEAR": IntentAction(
        parameter="beta",
        direction=-1,
        probe_delta=0.5,
        secondary={"tilt": (+1, 3.0)},
    ),
    "TOO_LOUD": IntentAction(parameter="output_gain", direction=-1, probe_delta=0.5),
    "TOO_QUIET": IntentAction(parameter="output_gain", direction=+1, probe_delta=1.0),
    "TOO_SHARP": IntentAction(parameter="tilt", direction=-1, probe_delta=8.0),
}

#: Valid range for each parameter. These MUST match the clamps applied inside
#: :class:`~intent_se.audio.wiener.ParametricWienerFilter.set_parameters`.
#:
#: The filter clamps whatever it is handed, so the audio path is safe either
#: way. Clamping here as well keeps the probe's *reported* state equal to what
#: the DSP actually applies -- otherwise a run of complaints in one direction
#: walks a parameter past its limit and the probe reports a value (a negative
#: gain floor, say) that the filter never used.
PARAM_LIMITS: dict[str, tuple[float, float]] = {
    "beta": (0.1, 4.0),
    "gain_floor": (0.0, 1.0),
    "output_gain": (0.1, 4.0),
    "tilt": (-12.0, 12.0),
}


@dataclass
class ParameterUpdate:
    """One probe step, with enough context to be logged and audited."""

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


class ParameterProbe:
    """Drives one DSP parameter at a time from classifier and severity output.

    Used to answer a single question: **does the parameter actually move, and
    can the effect be heard?** It is not a calibrated controller -- the step
    sizes carry no perceptual meaning. See the module docstring.

    Steps are **relative and cumulative**: each complaint nudges the current
    values rather than jumping to an absolute target, so a user can say "still
    too noisy" repeatedly and keep moving in that direction. Values are clamped
    to :data:`PARAM_LIMITS`, matching the filter's own clamps.

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
    >>> probe = ParameterProbe()
    >>> update = probe.apply("TOO_NOISY", severity=0.8, confidence=0.95)
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
        """Compute and record the parameter step for one classified sentence.

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

        # Linear severity -> magnitude, chosen for want of anything better.
        # Whether the true relationship is linear, saturating, or something
        # else entirely is precisely what the perceptual study must determine.
        self._nudge(action.parameter, action.direction, action.probe_delta, severity)
        for param, (direction, probe_delta) in action.secondary.items():
            self._nudge(param, direction, probe_delta, severity)

        return record(True)

    def _nudge(self, parameter: str, direction: int, probe_delta: float, severity: float) -> None:
        """Move one parameter by ``direction * probe_delta * severity``, then clamp.

        Adjustments are cumulative, so repeated complaints in the same direction
        would otherwise walk a parameter out of its valid range.
        """
        value = self.parameters[parameter] + direction * probe_delta * severity
        low, high = PARAM_LIMITS[parameter]
        self.parameters[parameter] = float(min(max(value, low), high))

    # ------------------------------------------------------------------

    def bind(self, enhancer) -> None:  # noqa: ANN001 - avoids a circular import
        """Push the current parameters into a :class:`~intent_se.audio.SpeechEnhancer`.

        This is the step that makes a probe audible: the enhancer picks the new
        values up on its next frame, without interrupting the stream. It clamps
        whatever it receives, so a wild classifier output cannot destabilise the
        audio path.
        """
        enhancer.set_parameters(**self.parameters)
