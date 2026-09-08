# Intention-Based Speech Enhancement for Hearing Aids Using NLP

Real-time speech enhancement whose DSP parameters are controlled by what the
user *says*, in plain language.

> BSc thesis, DTU (Electrical Engineering), in collaboration with Oticon.

---

## The problem

When something sounds wrong, a hearing aid user does not think in terms of gain
or noise-suppression strength. They think *"the background noise is too much"*
or *"voices sound muffled"*. The device, meanwhile, needs a technical
parameter — not a symptom description.

That mismatch is the gap this project closes. And it is getting harder to avoid:
as devices shrink toward completely-in-canal designs there is no room for
physical controls, so adjustment has to happen through an app or by voice.

**The question:** can natural language itself become the control interface?

## The system

```
                        ┌──────────────────────────────────────────┐
  "the background       │  NLP PIPELINE                            │
   noise is             │                                          │
   unbearable"  ───────►│  all-mpnet-base-v2  ──►  intent (6-way)   │
                        │  768-dim embedding  ──►  severity [0,1]   │
                        └────────────────────┬─────────────────────┘
                                             │
                                   ┌─────────▼──────────┐
                                   │ PARAMETER          │  ◄── calibration
                                   │ CONTROLLER         │      is open
                                   └─────────┬──────────┘
                                             │ β, gain, tilt
                        ┌────────────────────▼─────────────────────┐
  mic  ───────────────► │  AUDIO PIPELINE                          │ ──►  ear
  512 samples/callback  │                                          │
                        │  sliding buffer → sqrt-Hann → rFFT(2048) │
                        │  → IMCRA noise PSD                       │
                        │  → parametric Wiener (β)                 │
                        │  → irFFT → overlap-add                   │
                        └──────────────────────────────────────────┘
```

Three components, and the bridge between them:

| Component | What it does | Status |
|---|---|---|
| **Audio DSP** | Real-time enhancement with parameters adjustable while running | Built and validated |
| **Intent classifier** | Maps a complaint sentence to one of six classes | Built and validated |
| **Severity scorer** | Continuous 0–1 estimate of complaint intensity | Built and validated |
| **Parameter controller** | Turns (intent, severity) into DSP changes | **Directions grounded, magnitudes uncalibrated** |

That last row is the honest state of the project — see
[The open problem](#the-open-problem).

---

## Quick start

```bash
git clone https://github.com/oliburr10/Intention_based_speech_enhancment_using_NLP.git
cd Intention_based_speech_enhancment_using_NLP

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e '.[all]'
```

Verify the install:

```bash
pytest                          # 93 tests
intent-se-demo                  # scripted control loop, no hardware needed
```

Train the NLP models (downloads the 110M-parameter embedding model on first run):

```bash
intent-se-train --output-dir artifacts
```

Run the full system — live audio, steered by typed complaints:

```bash
intent-se-run --list-devices            # find your soundcard
intent-se-run --device 3
```

```
> the background noise is unbearable
    [applied] TOO_NOISY (severity 0.94, confidence 0.98) -> beta=1.940, gain_floor=0.012, ...
> now voices sound muffled
    [applied] SPEECH_UNCLEAR (severity 0.61, confidence 0.96) -> beta=1.635, tilt=1.830, ...
```

No soundcard? `intent-se-run --dry-run` exercises the NLP path and prints the
parameter decisions without opening a stream.

---

## Repository layout

```
├── data/
│   └── complaints_v4.csv        1106 labelled sentences (the dataset)
├── src/intent_se/
│   ├── config.py                every tunable, with its justification
│   ├── audio/
│   │   ├── stft.py              sliding buffer, sqrt-Hann, WOLA reconstruction
│   │   ├── imcra.py             noise PSD estimation + pre-roll seeding
│   │   ├── wiener.py            parametric Wiener filter (β is the NLP knob)
│   │   └── pipeline.py          the real-time enhancer
│   ├── nlp/
│   │   ├── dataset.py           loading, validation, stratified splitting
│   │   ├── embeddings.py        all-mpnet-base-v2 with on-disk caching
│   │   ├── classifier.py        five candidates + CV selection protocol
│   │   ├── severity.py          Ridge regression severity scorer
│   │   └── evaluate.py          confusion matrix, t-SNE, result tables
│   ├── control/
│   │   └── mapping.py           NLP → DSP bridge  ← the open problem
│   └── cli/
│       ├── train_nlp.py         reproduces the full selection protocol
│       ├── run_realtime.py      live interactive system
│       └── demo.py              offline control-loop demonstration
├── tests/                       93 tests
└── notebooks/                   original exploratory notebooks
```

---

## The audio pipeline

### Frame sizes, and why they are what they are

The chain starts from a hardware constraint and everything follows from it.

| Parameter | Value | Reason |
|---|---|---|
| Callback block | 512 | Imposed by the RME Fireface UC. Not a design choice. |
| Window `Nw` | 1024 | What the STFT needs for adequate frequency resolution. |
| Hop `H` | 512 | `= Nw/2`, exactly the 50% overlap WOLA requires. |
| FFT size | 2048 | Zero-padding, for finer bin interpolation. |
| Bins | 1025 | `n_fft/2 + 1`. |

Two size doublings happen here and they do completely different jobs. **512 → 1024**
is time-domain buffering: a sliding buffer accumulates enough samples for a full
window. **1024 → 2048** is frequency-domain zero-padding: it interpolates between
bins so IMCRA and the Wiener filter see a higher-resolution spectrum and produce
smoother gain curves. After the inverse FFT the padding is discarded.

### Perfect reconstruction

The analysis window is the *square root* of a periodic Hann window, applied
again at synthesis. Their product is one full Hann window, and at 50% overlap a
Hann window sums to exactly 1.0 across the two overlapping frames — the WOLA
condition. Get this wrong and overlapping contributions do not sum cleanly,
producing amplitude ripple that sounds like distortion.

The system self-tests this at startup:

```python
>>> from intent_se.audio import SpeechEnhancer
>>> SpeechEnhancer().self_test()
4.44e-16
```

Machine epsilon. The output is identical to the input apart from what is
deliberately done in the frequency domain.

### IMCRA noise estimation

The Wiener filter needs to know how much noise power sits in each bin *right
now*, and noise is never directly observable apart from speech. Two simpler
approaches fall short:

- A **hard VAD** makes a binary decision. When it is wrong the noise estimate
  either stalls during speech or is corrupted by speech energy.
- **Minimum statistics** tracks the per-bin minimum over a sliding window.
  Better, but it adapts slowly and cannot distinguish a genuinely rising noise
  floor from sustained speech.

IMCRA combines minimum tracking with a **soft per-bin speech-presence
probability**, which controls how fast the estimate adapts — quickly when speech
is unlikely, frozen when it is likely. Parameters follow Cohen (2003); the
minimum-search window is `D = U × V = 8 × 15 = 120` frames.

**The cold-start problem.** IMCRA is recursive, so at startup its buffers are
zero and the estimate is badly wrong — in practice the system suppressed
everything or nothing for the first few seconds. The fix is **pre-roll noise
floor seeding**: the first 5 frames are captured unfiltered, and their per-bin
*minimum* initialises the internal buffers. The minimum rather than the mean, so
a transient during the pre-roll biases the estimate low rather than high — an
estimate that starts too high suppresses speech from the first frame, which is
the more damaging failure.

### Parametric Wiener filter

The standard Wiener gain is `ξ/(1+ξ)`. Mathematically optimal, but with two
practical problems: high-frequency bins usually have lower SNR, so the filter
suppresses them hard and starts acting like a low-pass filter; and the curve is
fixed, offering no control over aggressiveness.

The parametric form introduces an exponent:

```
G(k) = ( ξ(k) / (1 + ξ(k)) ) ^ β
```

Low β flattens the curve and preserves high-frequency content; high β steepens
it and suppresses noise harder; **β = 1 recovers the standard filter exactly**.

**β is the parameter the NLP side controls.** Without a tunable parameter there
would be nothing for the classifier output to change.

A **gain floor of 5%** stops any bin being zeroed out completely — a bin fully
suppressed in one frame and back in the next is exactly what musical noise
("underwater", "bubbling") is.

The a priori SNR uses the Ephraim–Malah decision-directed estimator, which
blends the previous frame's clean-speech estimate with the current
maximum-likelihood one. A purely instantaneous estimate fluctuates wildly
between frames and the gain fluctuates with it.

---

## The NLP pipeline

### Dataset

No public dataset exists for hearing-aid complaints with severity labels, so one
was built: **1106 sentences, six classes**, LLM-generated with linguistic
diversity as an explicit goal.

| Class | Count | Meaning |
|---|---:|---|
| `TOO_NOISY` | 225 | Too much background noise |
| `SPEECH_UNCLEAR` | 194 | Cannot follow speech |
| `TOO_LOUD` | 175 | Overall volume too high |
| `TOO_QUIET` | 170 | Insufficient volume |
| `TOO_SHARP` | 167 | Harsh / shrill high frequencies |
| `BALANCED` | 175 | Satisfaction |

`BALANCED` earns its place: without it every utterance reads as a complaint and
the system keeps adjusting even when the user is perfectly happy.

Severity is continuous in `[0, 1]`, labelled by a four-level linguistic
heuristic — minimising language ("slightly", "barely") ≈ 0.05–0.25; intensive
("very", "constantly") ≈ 0.5–0.8; extreme ("unbearable", "cannot at all") ≈ 1.0.
`BALANCED` is always exactly 0.

The vocabulary of `TOO_LOUD` and `TOO_SHARP` is deliberately disjoint —
loudness words versus brightness words — because the two classes are the
easiest pair to confuse.

### Embeddings

`all-mpnet-base-v2` (768-dim) over `all-MiniLM-L6-v2` (384-dim). MiniLM is
smaller and faster, but mpnet scores higher on Sentence-BERT similarity
benchmarks, and that matters because several sentences sit near class
boundaries — *"I can't make out what people are saying"* could plausibly be
`TOO_NOISY` or `SPEECH_UNCLEAR`. Inference speed is not a bottleneck for a
proof-of-concept, so embedding quality wins. Vectors are L2-normalised, so
cosine similarity reduces to the dot product.

### Classifier selection

Five candidates are compared, spanning simple to complex. The spread answers a
real design question: **does the embedding space need non-linear boundaries, or
has the transformer already made the classes linearly separable?**

| Classifier | Family |
|---|---|
| Logistic regression | Linear |
| Linear SVM | Linear, max-margin |
| kNN (k=15, cosine) | Instance-based |
| MLP (2 ReLU layers) | Non-linear |
| XGBoost | Boosted trees |

The protocol:

1. **Five-fold stratified CV** on the development pool (train + val, 940
   sentences), scored by **macro-F1** — which weights all six classes equally,
   so a classifier gets no credit for doing well on the large classes while
   underperforming on the small ones.
2. The **test set is touched exactly once**, after selection. That single use is
   what makes the reported score an unbiased estimate.
3. Ties are broken on **generalisation, not raw accuracy**: the smaller the
   validation→test drop, the more trustworthy the model.

### Severity scorer

Two users both classified `TOO_NOISY` — one says *"a barely noticeable
background hum"*, the other *"absolutely excruciating"*. Same class, completely
different appropriate response. A fixed per-class adjustment either
under-responds to severe complaints or over-responds to mild ones.

**Ridge regression** on the same 768-dim embeddings, running in parallel with
the classifier. Ridge rather than OLS because with ~930 samples and 768
features, `N ≈ D`, and OLS in that regime fits the training data closely and
generalises poorly. Alpha is selected by CV on MAE.

> This component was **not** in the original project proposal. It was added
> after it became clear that purely categorical intent mapping was too rigid for
> a system meant to respond proportionally to the user's discomfort.

---

## Results

Reproduce with `intent-se-train --output-dir artifacts`, which writes tables to
`artifacts/tables/`, figures to `artifacts/figures/` and a `summary.json`.

Reported in the thesis:

| Classifier | CV macro-F1 | val→test drop |
|---|---:|---:|
| Logistic regression | 0.964 | 0.036 |
| **Linear SVM** ★ | **0.964** | **0.005** |
| MLP | 0.950 | 0.043 |
| XGBoost | 0.940 | — |
| kNN | 0.917 | — |

**Both linear classifiers beat all three non-linear ones.** That is the
substantive finding: the sentence transformer had already organised the six
classes into linearly separable regions, so a linear boundary is not merely
sufficient but *preferable* — it makes fewer assumptions and overfits less on a
dataset this size. The t-SNE projection shows the same thing from a different
angle.

Logistic regression and the SVM tied on CV, so the tie went to generalisation.
The SVM's validation→test drop of **0.005** means its CV estimate predicted its
test performance almost exactly. Logistic regression dropped 0.036; the MLP
dropped 0.043 despite the highest validation accuracy, which is what slight
overfitting looks like.

**Selected: linear SVM, test macro-F1 = 0.957.**

Severity scorer: **α = 0.10**, CV MAE **0.1002**. Best per class on `TOO_LOUD`
and `TOO_SHARP` (MAE ≈ 0.06); worst on `TOO_QUIET` (MAE ≈ 0.09), whose
sentences are too semantically similar across severity levels for the embedding
to separate mild from severe.

Where the classifier fails, it fails gracefully: `TOO_SHARP` is the main error
source, and a `TOO_SHARP` complaint handled as `TOO_LOUD` still moves the audio
in a perceptually reasonable direction.

> **Note on reproduction.** The exact figures above come from the thesis run.
> Re-running `intent-se-train` reproduces the protocol; small differences in
> the third decimal are expected across library versions and seeds.

---

## The open problem

**The parameter controller knows which direction to move. It does not know how
far.**

The direction of every adjustment is grounded in signal processing theory and
implemented:

| Intent | Response |
|---|---|
| `TOO_NOISY` | ↑ β — steepen the gain curve, suppress noise harder |
| `SPEECH_UNCLEAR` | ↓ β — flatten it, preserve high-frequency detail |
| `TOO_LOUD` | ↓ broadband output gain |
| `TOO_QUIET` | ↑ broadband output gain |
| `TOO_SHARP` | ↓ high-frequency tilt |
| `BALANCED` | decay back toward defaults |

This was confirmed to work: manually adjusting β during real-time operation
produces clearly audible changes in suppression, exactly as theory predicts. The
wiring is correct.

What is missing is the **magnitude**. If the severity scorer outputs 0.6 for one
complaint and 0.8 for another, how much larger should the second parameter
change be? Is the relationship linear? Does it saturate? There is no analytical
answer, because the right answer depends on how a real user perceives the
difference between two adjustment levels in their own acoustic environment.

Establishing it requires a structured listening study: recruit hearing-aid
users, present controlled variations of each parameter across the severity
range, have them rate which adjustments felt appropriate, and fit a mapping
function to the results. That is a research project in its own right.

**The `max_delta` values in `control/mapping.py` are therefore plausible
placeholders chosen to produce an audible but not destructive change — not
validated constants.** They are marked as such in the code.

---

## Limitations

1. **The perceptual calibration study above** — the largest gap.
2. **The dataset is LLM-generated.** Real complaints are messier and more varied
   than synthetic ones, and the severity labels reflect a single annotator's
   judgement rather than a standardised benchmark. Real-user ASR output will
   also be noisier than clean generated text ("peach sounds ruffled" instead of
   "speech sounds muffled").
3. **Compute.** `all-mpnet-base-v2` is 110M parameters and effectively wants a
   GPU. Fine for a laptop proof-of-concept, far beyond what embedded hearing-aid
   hardware supports. A real product needs a distilled model, or the NLP
   offloaded to a paired smartphone.
4. **The interface is typed text.** A real deployment needs ASR in front of the
   classifier.
5. **Mono processing.** Both microphone channels carry identical content and are
   averaged. A production system would keep them separate for binaural
   processing.

None of these require rethinking the core architecture.

---

## Development

```bash
pip install -e '.[dev]'
pytest                    # 93 tests
pytest --cov=intent_se    # with coverage
ruff check src tests      # lint
```

The test suite runs **fully offline** — no model download required. Tests that
would need the sentence transformer substitute a deterministic TF-IDF + SVD
projection, which exercises the pipeline plumbing without pretending to measure
embedding quality.

Notable tests:

- `test_stft.py` — WOLA condition holds at 50% overlap and *fails* at other hops
- `test_imcra.py` — the estimate does not absorb sustained speech energy
- `test_wiener.py` — β = 1 recovers the standard Wiener gain exactly
- `test_pipeline.py` — end-to-end SNR improvement on a synthetic mixture
- `test_control.py` — every adjustment direction is pinned down

## References

- Cohen, I. (2003). *Noise spectrum estimation in adverse environments: improved
  minima controlled recursive averaging.* IEEE Trans. Speech and Audio
  Processing, 11(5), 466–475.
- Ephraim, Y. & Malah, D. (1984). *Speech enhancement using a minimum
  mean-square error short-time spectral amplitude estimator.* IEEE Trans.
  ASSP, 32(6).
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence embeddings using
  Siamese BERT-networks.* EMNLP.
- Song, K. et al. (2020). *MPNet: Masked and permuted pre-training for language
  understanding.* NeurIPS.
- Loizou, P. (2013). *Speech Enhancement: Theory and Practice*, 2nd ed. CRC Press.

## License

MIT — see [LICENSE](LICENSE).
