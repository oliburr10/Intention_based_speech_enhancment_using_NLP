from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from intent_se.audio.pipeline import SpeechEnhancer
from intent_se.config import AudioConfig, NLPConfig
from intent_se.control.parameter_probe import ParameterProbe
from intent_se.nlp.classifier import IntentClassifier
from intent_se.nlp.embeddings import SentenceEmbedder
from intent_se.nlp.severity import SeverityScorer

BANNER = """\
Intention-Based Speech Enhancement -- interactive session
--------------------------------------------------------
Type a complaint in plain language, for example:
    the background noise is unbearable
    voices sound muffled
    everything is far too loud
    this sounds good now
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_realtime",
        description="Real-time speech enhancement controlled by natural language.",
    )
    parser.add_argument("--models", type=Path, default=Path("artifacts/models"),
                        help="Directory holding the trained .joblib models.")
    parser.add_argument("--device", type=_device, default=None,
                        help="Soundcard for BOTH input and output: an index "
                             "(e.g. 3) or a name substring (e.g. 'Fireface').")
    parser.add_argument("--input-device", type=_device, default=None,
                        help="Input device (the hearing aid microphone), if it "
                             "differs from --device.")
    parser.add_argument("--output-device", type=_device, default=None,
                        help="Output device (the headset), if it differs from "
                             "--device.")
    parser.add_argument("--samplerate", type=int, default=48_000)
    parser.add_argument("--blocksize", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip audio I/O; only print parameter decisions.")
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio devices and exit.")
    return parser


def _device(value: str) -> int | str:
    """Parse a device argument.

    sounddevice reads an ``int`` as a device index but a ``str`` as a
    case-insensitive substring match on the device *name*. Without this,
    argparse hands over ``"3"`` and sounddevice hunts for a device with "3" in
    its name instead of selecting index 3.
    """
    try:
        return int(value)
    except ValueError:
        return value


def _resolve(args) -> int | str | tuple[int | str, int | str] | None:  # noqa: ANN001
    """Combine the device arguments into what ``sd.Stream`` expects.

    Returns a single device when input and output are the same, a
    ``(input, output)`` tuple when they differ, or ``None`` for the system
    default.
    """
    inp = args.input_device if args.input_device is not None else args.device
    out = args.output_device if args.output_device is not None else args.device
    if inp is None and out is None:
        return None
    if inp == out:
        return inp
    return (inp, out)


def describe_devices(device) -> None:  # noqa: ANN001
    """Print the devices actually selected, so the routing is on the record."""
    try:
        import sounddevice as sd
    except ImportError:
        return
    pair = device if isinstance(device, tuple) else (device, device)
    for role, dev in zip(("input ", "output"), pair, strict=True):
        try:
            info = sd.query_devices(dev, "input" if role.strip() == "input" else "output")
            print(f"  {role}: [{info['index']}] {info['name']}  "
                  f"({info['max_input_channels']} in / "
                  f"{info['max_output_channels']} out, "
                  f"default {info['default_samplerate']:.0f} Hz)")
        except Exception as exc:
            print(f"  {role}: could not query {dev!r} -- {exc}")


def list_devices() -> int:
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is not installed. Install with: pip install '.[realtime]'")
        return 1
    print(sd.query_devices())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        return list_devices()

    #Load the NLP models
    classifier_path = args.models / "intent_classifier.joblib"
    scorer_path = args.models / "severity_scorer.joblib"

    for path in (classifier_path, scorer_path):
        if not path.exists():
            print(f"Missing model: {path}", file=sys.stderr)
            print("Train the models first:  python -m intent_se.cli.train_nlp",
                  file=sys.stderr)
            return 1

    print("Loading models ...")
    embedder = SentenceEmbedder(NLPConfig())
    classifier = IntentClassifier.load(classifier_path)
    scorer = SeverityScorer.load(scorer_path)
    probe = ParameterProbe()

    #Audio
    # The window is two blocks long: that is what gives the 50% overlap WOLA
    # needs. Deriving it here means --blocksize works for any value, instead of
    # only for its default.
    enhancer = SpeechEnhancer(
        AudioConfig(
            sample_rate=args.samplerate,
            block_size=args.blocksize,
            hop_length=args.blocksize,
            win_length=args.blocksize * 2,
            n_fft=args.blocksize * 4,
        )
    )
    print(f"WOLA self-test: max deviation {enhancer.self_test():.2e}")

    device = _resolve(args)

    stream_thread = None
    if not args.dry_run:
        print("Audio devices:")
        describe_devices(device)
        stream_thread = threading.Thread(
            target=enhancer.run_stream,
            kwargs={"device": device},
            daemon=True,
        )
        stream_thread.start()
        print(f"Audio stream running at {args.samplerate} Hz, "
              f"{args.blocksize}-sample blocks.")
    else:
        print("Dry run: no audio stream.")

    print(BANNER)

    # Interactive loop
    try:
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break

            if not text:
                continue
            if text in {":quit", ":q", ":exit"}:
                break
            if text == ":params":
                for key, value in sorted(probe.parameters.items()):
                    print(f"    {key:<12s} {value:.3f}")
                continue
            if text == ":reset":
                probe.reset()
                probe.bind(enhancer)
                print("    parameters restored to defaults")
                continue

            embedding = embedder.encode_one(text).reshape(1, -1)
            intent = classifier.predict_labels(embedding)[0]
            confidence = float(classifier.confidence(embedding)[0])
            severity = scorer.predict_one(embedding)

            update = probe.apply(intent, severity, confidence)
            probe.bind(enhancer)
            print("   ", update)

    except KeyboardInterrupt:
        pass

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
