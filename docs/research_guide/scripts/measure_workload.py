"""Measure the hardware-relevant properties of the thesis DSP pipeline.

Every number in Appendix A of the research guide comes from this script or is
derived from its output. Run it from the repository root:

    PYTHONPATH=src python docs/research_guide/scripts/measure_workload.py

It needs only numpy (the audio pipeline has no other dependencies).
"""

from __future__ import annotations

import time

import numpy as np

from intent_se.audio import SpeechEnhancer
from intent_se.config import AudioConfig

# Arrays that are per-bin but constant (computable or ROM), not mutable state.
CONSTANT_PER_BIN = {"_norm_freq"}


def walk_arrays(obj, prefix: str, seen: set[int]) -> list[tuple[str, tuple, int]]:
    """Recursively list every numpy array held by an intent_se object."""
    rows = []
    for name, value in vars(obj).items():
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            rows.append((f"{prefix}.{name}", value.shape, value.size))
        elif type(value).__module__.startswith("intent_se") and hasattr(value, "__dict__"):
            rows += walk_arrays(value, f"{prefix}.{name}", seen)
    return rows


def main() -> None:
    cfg = AudioConfig()
    fs, hop, win, n_fft, bins = (cfg.sample_rate, cfg.hop_length, cfg.win_length,
                                 cfg.n_fft, cfg.n_bins)
    frame_rate = fs / hop

    print("== Frame timing ==")
    print(f"fs={fs} Hz  hop={hop}  window={win}  n_fft={n_fft}  bins={bins}")
    print(f"frame rate   = {frame_rate:.4f} Hz")
    print(f"frame period = {1e3 / frame_rate:.4f} ms")
    print(f"algorithmic latency = {win} samples = {1e3 * win / fs:.3f} ms "
          "(one block of buffering + one hop of WOLA overlap)")
    for f_clk in (100e6, 50e6, 10e6, 1e6):
        print(f"  cycles per frame @ {f_clk / 1e6:>5g} MHz = {f_clk / frame_rate:>12,.0f}")

    # Run long enough that pre-roll has finished and every state array exists.
    enh = SpeechEnhancer()
    rng = np.random.default_rng(0)
    enh.process_signal(rng.standard_normal(hop * 40) * 0.05)

    print("\n== State held between frames ==")
    mutable_words = constant_words = tracker_words = 0
    other = 0
    for name, shape, size in walk_arrays(enh, "enh", set()):
        leaf = name.rsplit(".", 1)[-1]
        if size >= bins and size % bins == 0:
            per_bin = size // bins
            if leaf in CONSTANT_PER_BIN:
                constant_words += per_bin
                kind = "constant"
            else:
                mutable_words += per_bin
                kind = "mutable"
                if "_tracker" in name:
                    tracker_words += per_bin
            note = f"{per_bin} word(s)/bin, {kind}"
        else:
            other += size
            note = "buffer / kernel"
        print(f"  {name:46s} {str(shape):12s} {size:6d}  {note}")

    total = mutable_words * bins
    print(f"\nmutable words per bin      = {mutable_words}  "
          f"(of which {tracker_words} in the minimum trackers)")
    print(f"constant words per bin     = {constant_words}")
    print(f"mutable per-bin state      = {total:,} words")
    for bits in (16, 24, 32):
        print(f"  @ {bits} bit = {total * bits / 8 / 1024:.1f} KiB")
    print(f"other array words          = {other:,} (input buffer, OLA, window, kernel)")
    print(f"pre-roll buffer at startup = {cfg.preroll_frames} x {bins} = "
          f"{cfg.preroll_frames * bins:,} words (transient)")

    print("\n== Python float64 reference speed ==")
    enh = SpeechEnhancer()
    seconds = 10
    signal = rng.standard_normal(fs * seconds) * 0.05
    t0 = time.perf_counter()
    enh.process_signal(signal)
    elapsed = time.perf_counter() - t0
    n_frames = int(np.ceil(len(signal) / hop))
    print(f"{1e3 * elapsed / n_frames:.3f} ms per frame "
          f"(budget {1e3 / frame_rate:.3f} ms), real-time factor {elapsed / seconds:.3f}")

    print("\n== Alternative frame configurations (no zero padding, hop = window/2) ==")
    print("  fs [Hz]  window  bins  latency [ms]  frame rate [Hz]  state [words]")
    for fs_alt, w in ((48_000, 1024), (48_000, 512), (16_000, 256), (16_000, 128), (16_000, 64)):
        b = w // 2 + 1
        print(f"  {fs_alt:7d}  {w:6d}  {b:4d}  {1e3 * w / fs_alt:12.2f}  "
              f"{fs_alt / (w / 2):15.2f}  {mutable_words * b:13,d}")


if __name__ == "__main__":
    main()
