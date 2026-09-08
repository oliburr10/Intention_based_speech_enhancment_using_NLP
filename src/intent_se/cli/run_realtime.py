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
    parser.add_argument("--device", default=None,
                        help="Soundcard device index or name.")
    parser.add_argument("--samplerate", type=int, default=48_000)
    parser.add_argument("--blocksize", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip audio I/O; only print parameter decisions.")
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio devices and exit.")
    return parser


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
    enhancer = SpeechEnhancer(
        AudioConfig(sample_rate=args.samplerate, block_size=args.blocksize,
                    hop_length=args.blocksize)
    )
    print(f"WOLA self-test: max deviation {enhancer.self_test():.2e}")

    stream_thread = None
    if not args.dry_run:
        stream_thread = threading.Thread(
            target=enhancer.run_stream,
            kwargs={"device": args.device},
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
