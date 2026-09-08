# Intention-Based Speech Enhancement for Hearing Aids Using NLP

BSc thesis, DTU (Electrical Engineering) in collaboration with Oticon — Aron Burrell.

## The idea

Hearing aid users describe problems in symptom language ("voices sound muffled",
"the background noise is too much"), but the device needs a technical parameter.
This project closes that gap: natural language becomes the real-time control
interface for speech enhancement. Sentence in → DSP parameter change out.

## Three components

1. **Real-time audio DSP** — a speech enhancement pipeline with parameters that
   are adjustable while it runs.
2. **NLP intent classifier** — maps a complaint sentence to one of six intent
   classes.
3. **Severity scorer** — a continuous 0–1 estimate of how bad the complaint is,
   so the response is proportional rather than a fixed per-class step.

The bridge between them (the NLP→DSP parameter controller) is the open piece —
see "Status" below.

## Audio pipeline

- Soundcard: RME Fireface UC, **512 new samples per callback** (hardware
  constraint; everything else follows from it).
- **Sliding buffer** 512 → 1024: each callback shifts out the oldest 512 and
  writes in the new 512, so the STFT always sees the most recent 1024 samples.
  This also gives exactly 50% overlap (hop H = 512 = Nw/2), which WOLA requires.
- **FFT size 2048** (zero-padded from 1024) — finer bin interpolation, so IMCRA
  and the Wiener filter get a higher-resolution spectral view and smoother gain
  curves. Padding is discarded after the inverse FFT.
  *Keep the two doublings distinct:* 512→1024 is time-domain buffering,
  1024→2048 is frequency-domain zero-padding. Different jobs.
- **Window: sqrt-Hanning** applied at both analysis and synthesis. The product of
  the two square-root windows is one full Hanning window, which satisfies the
  WOLA condition (overlapping contributions sum to 1.0 at 50% overlap) and gives
  perfect reconstruction.
- **Mono averaging** — the hearing aid mic delivers two channels with identical
  content; they are averaged before the STFT and the processed mono output is
  routed to both ears. Proof-of-concept choice; a production system would keep
  channels separate for binaural processing.
- **Noise PSD estimation: IMCRA** (Improved Minima Controlled Recursive
  Averaging). Chosen over a hard VAD (brittle binary decision) and plain minimum
  statistics (adapts slowly, can't tell a rising noise floor from sustained
  speech). IMCRA combines minimum tracking with a soft per-bin speech-presence
  probability that controls adaptation speed.
  - **Pre-roll noise floor seeding** — IMCRA is recursive, so at startup its
    buffers are zero and the estimate is badly wrong for the first few seconds.
    Fix: collect the first 5 frames of background noise unfiltered, take the
    per-bin minimum, and use it to initialize IMCRA's buffers.
- **Parametric Wiener filter** with tunable exponent **β**. The standard Wiener
  gain over-suppresses high frequencies (lower SNR up there → acts like a
  low-pass filter) and offers no control over aggressiveness. β controls the
  steepness of the gain curve; β = 1 recovers the standard Wiener filter.
  **β is the knob the NLP side turns.**
- **Gain floor at 5%** — no bin is suppressed below 5% of its original level,
  which avoids musical-noise / "underwater" artifacts.

## NLP pipeline

- **Dataset: 1106 labeled sentences**, LLM-generated (no public dataset exists
  for hearing aid complaints with severity labels).
- **Six intent classes:** `TOO_NOISY`, `SPEECH_UNCLEAR`, `TOO_LOUD`,
  `TOO_QUIET`, `TOO_SHARP`, and `BALANCED`. `BALANCED` (satisfaction) matters —
  without it every input reads as a complaint and the system adjusts even when
  the user is happy.
- **Severity labels 0–1**, four-level linguistic heuristic: minimizing language
  ("slightly", "barely") ≈ 0.05–0.25; intensive ("very", "constantly")
  ≈ 0.5–0.8; extreme ("unbearable", "cannot at all") ≈ 1.0; `BALANCED` = exactly 0.
- **Embeddings: `all-mpnet-base-v2`** (Sentence-BERT family, MPNet backbone,
  768-dim, L2-normalized → cosine similarity is the right metric). Chosen over
  `all-MiniLM-L6-v2` (384-dim, faster) because borderline sentences
  ("I can't make out what people are saying" → `TOO_NOISY` or `SPEECH_UNCLEAR`?)
  need the better semantic placement; inference speed is not a bottleneck here.
- **Classifiers evaluated (5):** logistic regression, linear SVM, kNN (k=15,
  cosine, weighted vote), MLP (2 ReLU hidden layers), XGBoost. The point of the
  spread was a real design question: is the embedding space already linearly
  separable, or does it need non-linear boundaries?
- **Severity scorer: Ridge regression** on the same 768-dim embeddings. Ridge
  over OLS because N ≈ D (~930 samples vs 768 features) → OLS overfits.
  α selected by 5-fold CV over five candidates; **α = 0.10**, CV MAE **0.1002**.
  Note in the write-up that the severity scorer was an *addition* beyond the
  original project proposal.

## Evaluation and results

- **Split:** 70% train; remaining 30% split evenly into validation and test
  (~166 each, ~26 per class). 5-fold stratified CV on the combined train+val
  pool of 940. Test set used exactly once, after selection.
- **CV mean macro-F1:** logistic regression and linear SVM **tied at 0.964**;
  MLP 0.950; XGBoost 0.940; kNN 0.917.
- **Key pattern:** both linear classifiers beat all three non-linear ones — the
  sentence transformer already organized the classes into linearly separable
  regions. This validates the embedding-model choice.
- **Tie broken on generalization, not raw accuracy.** val→test macro-F1 drop:
  SVM **0.005** (0.962 → 0.957); logistic regression 0.036 (0.975 → 0.939);
  MLP 0.043 (highest val accuracy 0.982, so it overfit slightly).
  **SVM selected. Test macro-F1 = 0.957.**
- **Confusion matrix:** `TOO_NOISY`, `TOO_LOUD`, `TOO_QUIET`, `BALANCED`
  essentially perfect; `SPEECH_UNCLEAR` two errors; `TOO_SHARP` is the main
  error source. Low-stakes failure — a `TOO_SHARP` complaint read as `TOO_LOUD`
  still triggers an adjustment in a perceptually reasonable direction.
- **t-SNE:** five of six classes well separated; `TOO_LOUD` / `TOO_SHARP`
  overlap in the lower-left. ⚠️ Known inconsistency between two drafts of the
  presentation: one says `TOO_SHARP`'s errors go to `TOO_LOUD`, the other says
  they go mostly to `SPEECH_UNCLEAR`. **Check the actual confusion matrix before
  repeating either claim.**
- **Severity scorer per class:** best `TOO_LOUD` / `TOO_SHARP` (MAE ≈ 0.06);
  worst `TOO_QUIET` (MAE ≈ 0.09) — its sentences are too semantically similar
  across severity levels.

## Status

- ✅ Audio pipeline — built and validated (perfect WOLA reconstruction, stable
  IMCRA from a cold start, audibly adjustable suppression).
- ✅ NLP classifier + severity scorer — built and validated on held-out data.
- ⚠️ NLP→DSP parameter mapping — **architecture in place, calibration open.**
  The *direction* of every adjustment is theoretically grounded and implemented
  (`TOO_NOISY` ↑β, `SPEECH_UNCLEAR` ↓β, `TOO_LOUD` ↓gain, `TOO_QUIET` ↑gain).
  The *magnitude* is not analytically derivable — how much β change a severity
  of 0.6 vs 0.8 should produce depends on user perception, which needs a
  structured listening study. Verified manually that adjusting β in real time
  produces clearly audible change, so the wiring works; only the calibration is
  missing.

## Limitations / future work

1. Perceptual calibration study for the severity→magnitude mapping (the big one).
2. Dataset from real users — LLM-generated language doesn't match how users
   actually speak, and severity labels are subjective.
3. Lighter-weight embedding model — `all-mpnet-base-v2` is 110M params and wants
   a GPU; hearing aid hardware can't host it. Distill it, or offload NLP to a
   paired smartphone.
4. ASR front-end — the current interface is typed sentences in a terminal.

None of these require rethinking the core architecture.

## Outstanding thesis-document fixes

From supervisor feedback (Sigtryggur):

- Nomenclature section (page III) is full of wrong EV/AC terms.
- Blank placeholders in §2.5.5 and Ch. 4 should read "six".
- Add the GitHub link in the preface.
- Numbering: tables skip 2, figure labels are messy, and the equation reference
  in §2.3.3 should be 14, not 13.
- Discussion should call out: real-user ASR errors (LLM data is too clean),
  whether SBERT embeddings capture meaning or just surface patterns, the
  noise-floor problem if the user activates while someone is speaking, the Ridge
  score→parameter mapping, and using a Euclidean KD-tree for kNN (embeddings are
  already normalized).

## Where the code lives

- `src/intent_se/audio/` -- `stft.py` (WOLA), `imcra.py`, `wiener.py`, `pipeline.py`
- `src/intent_se/nlp/` -- `dataset.py`, `embeddings.py`, `classifier.py`,
  `severity.py`, `evaluate.py`
- `src/intent_se/control/mapping.py` -- the NLP->DSP bridge (the open problem;
  its `max_delta` values are uncalibrated placeholders, marked as such)
- `src/intent_se/config.py` -- every tunable, with its justification in the docstring
- `data/complaints_v4.csv` -- the 1106-sentence dataset
- `tests/` -- 93 tests, runs fully offline (no model download)

Entry points: `intent-se-train`, `intent-se-run`, `intent-se-demo`.

The original Jupyter notebooks are **not** in this repo or in Drive; `notebooks/`
holds a placeholder README for them.

## Conventions

- Python: PyTorch, NumPy, SciPy, librosa/torchaudio, scikit-learn, XGBoost,
  sentence-transformers.
- Metric names to use precisely: macro-F1 for classification, MAE for severity.
- Be careful with the term "perfect reconstruction" — it holds for the WOLA
  analysis/synthesis path, not for the enhanced output.
