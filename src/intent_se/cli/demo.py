"""Offline demonstration of the control loop, with no audio hardware needed.

Runs a scripted sequence of complaints through the controller and prints the
resulting DSP parameters, so the language-to-parameter behaviour can be
inspected without a soundcard or trained models.

Usage
-----
::

    python -m intent_se.cli.demo
"""

from __future__ import annotations

from intent_se.control.mapping import ParameterController

SCRIPT: list[tuple[str, str, float]] = [
    ("there is a bit of background noise", "TOO_NOISY", 0.20),
    ("the background noise is unbearable", "TOO_NOISY", 0.95),
    ("now voices sound muffled", "SPEECH_UNCLEAR", 0.60),
    ("everything is far too loud", "TOO_LOUD", 0.80),
    ("the treble is painfully shrill", "TOO_SHARP", 0.75),
    ("this sounds good now", "BALANCED", 0.00),
]


def main() -> int:
    controller = ParameterController()

    print("Scripted control-loop demonstration")
    print("=" * 78)
    print(f"defaults: {controller.parameters}\n")

    for sentence, intent, severity in SCRIPT:
        update = controller.apply(intent, severity, confidence=0.95)
        print(f'"{sentence}"')
        print(f"    {update}\n")

    print("=" * 78)
    print("Reminder: adjustment *directions* are theoretically grounded, but the")
    print("*magnitudes* are uncalibrated placeholders awaiting a perceptual study.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
