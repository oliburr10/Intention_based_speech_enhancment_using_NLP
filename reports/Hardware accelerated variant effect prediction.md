# Hardware-accelerated variant effect prediction for pharmacogenomics

A research brief on whether — and how — to build an FPGA/ASIC project around predicting
whether protein mutations affect drug response.

Prepared 17 October 2026. All figures carry sources; conflicting or unverified numbers are
flagged as such.

---

## 1. The headline finding

Two independent gaps make this project viable, and they reinforce each other:

**Gap 1 (science).** General-purpose variant effect predictors demonstrably fail on
pharmacogenes, and the failure has now been quantified. A 2026 *G3* paper that deep-mutationally
scanned CYP2C9, CYP2C19 and NUDT15 found:

- **28%** of CYP2C9 variants (1,236 of 4,421) *decouple catalytic activity from protein
  abundance* — the protein is present but dead, or absent but active.
- **48%** of NUDT15 variants (1,364 of 2,844) decouple thiopurine sensitivity from stability.
- **AlphaMissense scored 38 of 195 "stable-but-dead" CYP2C9 variants as likely benign.**

The authors' conclusion is blunt: pharmacogene interpretation requires assay-specific,
substrate-aware functional data, not a single generic pathogenicity score. That is a documented
hole in the state of the art, with a number attached to it — exactly what a thesis needs.

**Gap 2 (hardware).** There appears to be **no FPGA or ASIC accelerator for protein language
model inference or in-silico saturation mutagenesis.** Targeted searches across FPGA venues and
arXiv returned only general transformer/LLM accelerators and only GPU/CPU deployment for ESM-2.
Treat this as "no prior art found in a reasonable search," not as proof of absence — but it is a
strong signal, and confirming it properly is itself a thesis contribution (see §7).

The combination is the project: **assay-aware pharmacogene variant prediction, with a hardware
engine that makes it cheap enough to run at scale.**

---

## 2. State of the art in variant effect prediction

### Where the field stands

| Model | Inputs | Notes |
|---|---|---|
| **AlphaMissense** (DeepMind, 2023) | Structure + sequence | Scored ~71M human missense variants; precomputed scores released. Mean ProteinGym Spearman **0.514** |
| **ESM-1v / ESM-2** | Sequence only | Zero-shot, no MSA needed. ESM-2 is encoder-only, 6–72 layers, hidden 320–5,120 — the 8M/35M sizes are the realistic hardware targets |
| **EVE / Tranception / TranceptEVE** | MSA-based | Strong but MSA construction dominates runtime |
| **VenusREM2** (2026) | Sequence + retrieval | Reports official ProteinGym Spearman **0.556** for a 6-member ensemble, vs **0.518** for the strongest listed baseline |
| **Kermut**-based entries | Supervised | A third-party leaderboard lists **0.681**, Kermut itself at 0.633 — *unverified, see warning below* |
| General-purpose LLMs | Sequence as text | Top out around **0.41** on the PG-LLM benchmark — specialized models remain clearly ahead |

> ⚠️ **Verify the leaderboard yourself.** The three sources above give 0.514, 0.556 and 0.681 as
> "top" numbers. They disagree because of different assay subsets, aggregation rules and
> verification standards. Go to `proteingym.org` directly and read its aggregation rules before
> quoting any figure in a thesis proposal.

### The scoring-strategy detail that decides your hardware story

This is the single most important technical point in this brief, and it cuts against the
obvious framing. How you score variants changes the compute cost by orders of magnitude:

| Strategy | Forward passes | Cost to score all variants of one protein |
|---|---|---|
| **Wildtype marginal** | **1 per protein** | Read all 19×L substitution scores off one pass's logits |
| **Masked marginal** | **L per protein** (mask each position) | ~500 passes for a 500-residue protein |
| **Mutant marginal / pseudo-likelihood** | 1 per *variant* | ~9,500 passes |

**Why this matters:** the tempting pitch is "71 million variants, therefore hardware." But with
wildtype-marginal scoring and a sequence-only model, the whole human proteome is roughly
**20,000 forward passes** — minutes on one GPU. That pitch collapses under scrutiny, and a
sharp examiner will say so.

The honest hardware motivations are these instead, in descending order of strength:

1. **Masked-marginal at proteome scale** — ~10M forward passes, which is a real workload, and
   masked marginal generally scores better than wildtype marginal.
2. **MSA construction** (for EVE/GEMME-class methods) — this is the actual bottleneck in
   MSA-based pipelines, and HMMER/JackHMMER is a classic, well-matched hardware target.
3. **Energy per prediction as a first-class metric** — not just speed. This is the framing where
   custom silicon always wins and where your background is strongest.
4. **Embedded / point-of-care genotyping** — a bedside pharmacogenomic test, where there is no
   GPU and the power budget is fixed.
5. **Delta/incremental inference** — the 19 substitutions at a position share nearly the whole
   sequence. Whether hardware can exploit that reuse is an open question I found no literature
   on.

Pick your motivation deliberately and defend it in the first chapter. Do not hand-wave "71
million variants."

---

## 3. Datasets: what ground truth actually exists

### Pharmacogenes with deep mutational scanning

| Gene | Drug relevance | DMS status |
|---|---|---|
| **CYP2C9** | Warfarin | 4,421 variants, activity + abundance (2026 *G3*) |
| **CYP2C19** | Clopidogrel | Abundance scan; found a substrate specificity–abundance tradeoff. K′ helix variants deleterious in CYP2C19 but tolerated in CYP2C9 |
| **NUDT15** | Thiopurines | 2,844 variants, thiopurine sensitivity + stability |
| **TPMT** | Thiopurines | VAMP-seq (abundance) |
| **VKORC1** | Warfarin | VAMP-seq + a VKOR activity scan |

The CYP2C9/CYP2C19/NUDT15 trio is the best target: recent, large, and — critically — it has
**two readouts per variant** (activity *and* abundance). That lets you pose multi-task
prediction, which is where the scientific novelty lives.

### Supporting resources

- **MaveDB**, **ProteinGym** — benchmark aggregation
- **PharmVar**, **PharmGKB/ClinPGx**, **CPIC** — allele definitions and dosing guidelines
- **ClinVar**, **gnomAD** — clinical annotation and population frequency
- **CIViC**, **OncoKB** — cancer drug-response biomarkers
- **AlphaFold DB**, **PDB** — structures, including distance-to-active-site features
- **FDA Table of Pharmacogenomic Biomarkers in Drug Labeling** — the clinically actionable set

### A Danish dataset angle worth knowing

Kresten Lindorff-Larsen's group at the **University of Copenhagen** published a 2026 *Protein
Science* paper (Schulze, Blaabjerg, Cagiada, Lindorff-Larsen; doi `10.1002/pro.70526`) training
a supervised abundance predictor on **~32,000 VAMP-seq variant scores**, with dataset-specific
standard curves so independent experiments can be pooled. They also have a March 2026 bioRxiv
preprint analysing where ten variant effect predictors *disagree* across 13,000+ human proteins,
and using that disagreement to prioritise experiments.

That second paper is a gift: predictor disagreement is a ready-made way to choose which proteins
your hardware should spend its cycles on.

---

## 4. Hardware: prior art and what it tells you

### Transformer inference on FPGAs — the relevant reference points

- **HLSTransform**: int8, written in high-level synthesis, **57.11 tokens/s for a 110M-parameter
  model** on a Xilinx Virtex UltraScale+ VU9P. Proof that HLS is a viable route for a student and
  that ~100M parameters is tractable.
- **W4A8 quantization on an encoder (BERT-class) model**: **56.76% CoLA accuracy vs 56.84% at
  fp16.** ESM-2 is also encoder-only, so this is the closest architectural analogue available —
  and it suggests 4-bit weights will cost you very little. This is the key de-risking datapoint
  for your quantization phase.
- **2:4 structured sparsity + int4** on a systolic-array FPGA: up to **4× weight-storage
  reduction, 1.71× matmul speedup**.
- **LUT-LLM** (Nov 2025): first FPGA accelerator to deploy a 1B+ model using vector quantization
  and memory-based compute, **1.10–3.29× faster than GPUs**.

### Genomics acceleration — crowded, so differentiate carefully

- **Illumina DRAGEN** — the commercial FPGA incumbent.
- **Pair-HMM on FPGA**: 14.85× throughput over an 8-core CPU and **147.49× throughput per unit
  energy** (FPL 2017); an open-source AWS F1 implementation reached **15× in-pipeline with GATK**.
- **28nm end-to-end genome analysis ASIC** (IEEE TBCAS vol. 19, 2025): full pipeline in **28.2
  minutes average** on the precisionFDA dataset, 99.79% precision / 99.03% sensitivity, and
  reports higher energy efficiency than DRAGEN.

The lesson: **variant *calling* acceleration is a mature, competitive field.** Variant *effect
prediction* acceleration appears empty. Go where the field is empty.

### Toolchain reality check

- **hls4ml**: multi-head attention support landed in **v1.2.0**, via the HGQ2 front end, **Vitis
  backend only**. But an April 2026 GitHub discussion asking about QKV/`torch.matmul` support sits
  unanswered, and a September 2026 post reports flip-flop usage exceeding 100% with
  hls4ml+Brevitas on a KV260. Sources also conflict on how general the transformer support is.
  **Budget real time for tool friction, and prototype the attention block early.**
- **FINN**: no confirmed quantized-attention support found. Brevitas exports to FINN for
  integer-only dataflow, but attention specifically is unaddressed.
- **Fallback**: hand-written RTL (SystemVerilog or Chisel) for the matmul/softmax datapath. Less
  glamorous, more predictable, and at DTU the Chisel expertise is right there (§6).

---

## 5. ASIC tapeout: a time-sensitive window

If you want actual silicon, the calendar matters more than the design:

- **Tiny Tapeout IHP 26b** closed **21 September 2026** — about four weeks before this brief.
  You missed it.
- **Tiny Tapeout IHP 26a** closed March 2026; **chips expected February 2027.**
- Process is **IHP SG13G2, 130nm BiCMOS, open-source PDK**, hardened with the open-source flow.

So the realistic target is the **next shuttle (~early 2027)**, which fits a thesis that starts
now. Verify the current window at `tinytapeout.com` before planning around it — shuttle
schedules move.

**Two constraints to internalise:**

1. **A tile is tiny.** A transformer will not fit. One quantized MAC tile, a softmax unit, or a
   requantization block is the right scope.
2. **The obvious block is taken.** There is already a Tiny Tapeout IHP project implementing a
   weight-stationary int8 systolic MAC array with per-channel requantization. If you tape out
   "an int8 NPU tile," you have built someone's demo. Your tile needs to do something specific to
   *this* problem — the amino-acid embedding lookup, a 19-substitution delta-scoring datapath, or
   a fused softmax/log-odds unit that emits variant scores directly.

Also check **IHP's free university MPW runs** and **Europractice** access through DTU, which may
offer more area than a Tiny Tapeout tile.

---

## 6. People to talk to

### Hardware side, at DTU Compute

- **Martin Schoeberl** — professor, DTU Compute. Author of the Chisel book, with a preprint on
  open-source chip design and a paper on *bringing chip tapeout into university education*. If
  anyone at DTU can get a student to silicon, it is him. **Start here.**
- **Luca Pezzarossa** — associate professor, DTU Compute. Reconfigurable systems, computer
  architecture, **FPGA acceleration**. The natural day-to-day supervisor for the FPGA phase.
- **Jan Madsen** — Embedded Systems Engineering section contact.
- *Note:* **Jens Sparsø** is listed as **professor emeritus**, so not a likely primary supervisor.

A signal worth noting: DTU Compute has advertised a **PhD scholarship in open-source EDA tools**
and a professorship in Computer Engineering & Hardware Design (deadline January 2026). The
department is actively investing in exactly this area — good timing for you.

### Biology side

- **Kresten Lindorff-Larsen (University of Copenhagen)** — the obvious co-supervisor. His group
  *is* the Danish centre of gravity for variant effect and protein abundance prediction (§3).
  Cross-institution co-supervision takes paperwork but transforms the project's credibility.
- **DTU Health Tech bioinformatics** — closer to home administratively; worth asking who works on
  protein ML there.
- *Note:* Thea Schulze, first author on the 2026 abundance papers, has moved to the MRC LMB.

**Practical advice:** approach the hardware supervisor with the biology already scoped, and the
biology supervisor with the hardware already scoped. Neither will fill in the other half for you,
and a project that visibly spans both is what makes it memorable.

---

## 7. The recommended plan

**Working title:** *Assay-aware prediction of pharmacogene variant function, and hardware
acceleration of its inference*

The structure deliberately front-loads the science, so that **the project still succeeds if the
silicon does not.**

### Phase 0 — Scoping (2 weeks)

- Read the 2026 *G3* CYP2C9/CYP2C19/NUDT15 paper properly. It is the spine of your motivation.
- Pull the authoritative ProteinGym leaderboard and its aggregation rules.
- Confirm the Tiny Tapeout shuttle window and whether a Europractice/IHP route is open to you.
- Email Schoeberl and Pezzarossa. Email Lindorff-Larsen's group.

**Deliverable:** a two-page proposal with one quantified gap and one named metric.

### Phase 1 — Software baseline (5–7 weeks) · *the scientific core*

- Score the pharmacogene DMS datasets with ESM-2 (several sizes) using wildtype-marginal **and**
  masked-marginal strategies. Report Spearman per gene and per readout.
- Compare against precomputed AlphaMissense scores. **Try to reproduce the paper's finding** that
  general predictors mislabel stable-but-dead variants.
- Add structure features from AlphaFold DB: burial, distance to heme/active site, distance to the
  substrate pocket.
- Pose it as **multi-task**: predict activity and abundance separately rather than one fused
  "pathogenicity" score. This is where your novelty is, and it follows directly from the
  activity/abundance decoupling finding.

**Metric:** Spearman per gene per readout, plus accuracy specifically on the decoupled
(stable-but-dead) subset — the cases general predictors get wrong.

**Deliverable:** a result that stands alone as a thesis chapter even if no hardware is built.

### Phase 2 — Hardware-aware model (3–4 weeks)

- Quantize to int8 and int4 (Brevitas or QKeras), with quantization-aware training where needed.
- Plot **accuracy retention vs bit width vs model size.** The W4A8 encoder result in §4 suggests
  int4 will cost you very little — confirming that *on a protein model* is a publishable result
  in itself.
- Profile the model: which layers dominate FLOPs, what the on-chip memory footprint is, and what
  arithmetic the attention block actually needs.

**Deliverable:** the accuracy/precision trade-off curve, and a chosen configuration with a
defensible justification.

### Phase 3 — FPGA accelerator (6–10 weeks) · *the engineering core*

- Build a quantized encoder inference engine: systolic matmul array, softmax and GELU units,
  on-chip weight storage for the small ESM-2 sizes.
- Route: hls4ml if its attention support holds up for you; otherwise hand-written RTL. **Decide
  this by week 2 of the phase**, after a spike on the attention block alone.
- Verify with Verilator and cocotb against the PyTorch model, bit-exactly where possible.
- **Research angle (the differentiator):** exploit the structure of saturation mutagenesis. The
  19 substitutions at a position share nearly the entire sequence. Can the hardware cache and
  reuse activations across them? I found no literature on this, in software or hardware.

**Metrics:** variants/second, **joules per variant**, latency, and Spearman retained — always
against both a CPU and a GPU baseline, measured not quoted.

**Deliverable:** a working accelerator with measured numbers on real silicon-adjacent hardware.

### Phase 4 — Optional ASIC tile (4–6 weeks, if a shuttle is open)

- Tape out **one block**, not the model: the amino-acid embedding lookup, a delta-scoring
  datapath, or a fused softmax/log-odds unit that emits variant scores directly.
- Flow: OpenLane 2 / OpenROAD on IHP SG13G2.
- Report post-synthesis area, power and timing even if fabrication lands after submission.

**Deliverable:** a hardened GDS and a results table. Chips arrive roughly a year later, so this
is a bonus, not a dependency.

### Hardware shopping list

| Board | Fit |
|---|---|
| **PYNQ-Z2** | Learning only; too small for ESM-2 |
| **Kria KV260** | Good entry point; note the reported hls4ml resource pressure |
| **ZCU104** | Solid middle ground for ESM-2 8M/35M |
| **Alveo U50/U55C** | HBM, best for throughput claims; check DTU's inventory first |
| **Versal VCK190** | AI Engines; steeper learning curve |

Ask DTU Compute what they already own, and check the **AMD University Program** before buying
anything. Cloud FPGAs (AWS F1/F2) are a fallback — the open-source pair-HMM work ran there.

---

## 8. Risks, honestly stated

| Risk | Severity | Mitigation |
|---|---|---|
| **The throughput argument collapses** (wildtype-marginal scoring is cheap — §2) | **High** | Choose your motivation deliberately: masked-marginal scale, J/variant, or embedded deployment. Address it in chapter 1, don't let an examiner raise it first |
| hls4ml attention support is immature | High | Spike on attention in week 1 of Phase 3; keep hand-written RTL as the fallback |
| Pharmacogenes are poorly conserved, so PLMs may do badly | Medium | This is a *finding*, not a failure — and it is the paper's own point. Report it either way |
| Tapeout window missed or slips | Medium | Phase 4 is explicitly optional; FPGA results carry the thesis |
| Scope is genuinely two theses | **High** | Phase 1 must stand alone. Agree milestones with both supervisors up front |
| No prior art found ≠ no prior art exists | Medium | Do a proper systematic search (FCCM, FPGA, FPL, ISCA, MICRO, DAC, ICCAD) and write it up as a related-work section. If something exists, you want to find it in month 1, not month 6 |
| Overclaiming clinical relevance | Medium | See below |

### On the word "biomarker"

A validated clinical biomarker requires analytical and clinical validation — patient data,
regulatory process, the FDA's pharmacogenomic biomarker table. A computational project
**nominates candidates** and **prioritises experiments**. Phrase every claim that way, add a
disclaimer that outputs are not medical advice, and you will be both accurate and more
credible. The predictor-disagreement work in §3 is a good model for how to frame computational
output as experimental prioritisation rather than clinical conclusion.

---

## 9. Verdict

**Do it, with the science leading.** The project is real, the gap is documented with numbers, and
the hardware territory appears genuinely unoccupied.

Three things decide whether it succeeds:

1. **Make Phase 1 stand alone.** A multi-task pharmacogene predictor that beats AlphaMissense on
   the decoupled-variant subset is a thesis. Everything after it is upside.
2. **Pick an honest hardware motivation.** Energy per variant and embedded deployment are
   defensible. Raw throughput against a sequence-only model is not.
3. **Get the biology co-supervisor.** The Copenhagen group works on precisely this. Without that
   half, you will be an EE student guessing at biochemistry; with it, you are doing something
   nobody at either institution is doing.

The through-line to your existing work is worth stating in your proposal: your BSc thesis took a
model and asked how it runs in a constrained real-time system. This asks the same question of a
protein model. That is a coherent research identity, not a change of field.

---

## Sources

**Variant effect prediction**
- [Accurate proteome-wide missense variant effect prediction with AlphaMissense](https://www.science.org/doi/10.1126/science.adg7492) — *Science*
- [ProteinGym substitution benchmark leaderboard (third-party aggregation)](https://www.sota2.com/research/sota/protein-variant-effect-prediction-on-proteingym-substitution-benchmark)
- [A General Harness for Protein Foundation Model Fitness Prediction](https://arxiv.org/html/2609.34654)
- [PG-LLM: Benchmarking General-Purpose Language Models for Protein Variant Ranking](https://www.biorxiv.org/content/10.64898/2026.07.27.741045v3)
- [Scaling and Data Saturation in Protein Language Models](https://arxiv.org/pdf/2507.22210)
- [ESM-2 Protein Language Model (architecture overview)](https://www.emergentmind.com/topics/protein-language-model-esm-2)
- [Structure, Function, and Interaction in Protein Language Models (MIT thesis, 2025)](https://dspace.mit.edu/bitstream/handle/1721.1/159126/zheng-redzheng-meng-eecs-2025-thesis.pdf?sequence=1&isAllowed=y)
- [Residue-Level Attributions in Protein Language Models Do Not Recover Allergen Epitopes](https://arxiv.org/pdf/2606.22181)
- [A Comprehensive Review of Transformer-based Language Models for Protein Sequence Analysis and Design](https://arxiv.org/pdf/2507.13646)

**Pharmacogene datasets**
- [Deep mutational scanning of CYP2C9, CYP2C19, and NUDT15 shows that pharmacogene variant interpretation requires assay-specific functional data](https://academic.oup.com/g3journal/advance-article/doi/10.1093/g3journal/jkag248/8776100) — *G3*
- [Deep mutational scanning of CYP2C19 in human cells reveals a substrate specificity-abundance tradeoff](https://academic.oup.com/genetics/article/228/3/iyae156/7774808) — *Genetics*
- [Massively parallel characterization of CYP2C9 variant enzyme activity](https://www.biorxiv.org/content/10.1101/2021.03.12.435209v1.full.pdf) — bioRxiv
- [Measuring Pharmacogene Variant Function at Scale Using Multiplexed Assays](https://www.annualreviews.org/content/journals/10.1146/annurev-pharmtox-032221-085807) — *Annual Reviews*
- [Amorosi dissertation, University of Washington (2020)](https://digital.lib.washington.edu/bitstreams/c58dfdfe-5ecb-403a-a883-a108f8ac0c8a/download)

**Danish variant-effect work**
- [Supervised learning of protein variant effects across large-scale mutagenesis datasets](https://onlinelibrary.wiley.com/doi/10.1002/pro.70526) — *Protein Science*, 2026
- [Effects of residue substitutions on the cellular abundance of proteins](https://elifesciences.org/articles/103721) — *eLife*
- [Disagreement among variant effect predictors guides experimental prioritization of target proteins](https://www.biorxiv.org/content/10.64898/2026.03.18.712765v1) — bioRxiv
- [Decoding molecular mechanisms for loss-of-function variants in the human proteome](https://elifesciences.org/reviewed-preprints/108160) — *eLife*

**FPGA / transformer acceleration**
- [HLSTransform: Energy-Efficient Llama 2 Inference on FPGAs via High Level Synthesis](https://arxiv.org/pdf/2405.00738)
- [LUT-LLM: Efficient Language Model Inference with Memory-based Computations on FPGAs](https://arxiv.org/html/2511.06174v2)
- [FPGA Co-Design for Efficient N:M Sparse and Quantized Model Inference](https://arxiv.org/pdf/2512.24713)
- [Large Language Model Inference Acceleration: A Comprehensive Hardware Perspective](https://arxiv.org/pdf/2410.04466)
- [Understanding the Potential of FPGA-based Spatial Acceleration for LLM Inference](https://dl.acm.org/doi/10.1145/3656177)
- [Recent Developments in Transformer Inference Deployment on FPGA Platforms: A Survey](https://arxiv.org/pdf/2609.01212)
- [hls4ml: A Flexible, Open-Source Platform for Deep Learning Acceleration on Reconfigurable Hardware](https://arxiv.org/pdf/2512.01463)
- [hls4ml GitHub Discussions](https://github.com/fastmachinelearning/hls4ml/discussions)
- [Benchmarking Quantized Neural Networks on FPGAs with FINN](https://arxiv.org/pdf/2102.01341)

**Genomics acceleration**
- [On Accelerating Pair-HMM Computations in Programmable Hardware](https://ssbaner2.cs.illinois.edu/publications/fpl2017/) — FPL 2017
- [Open-source FPGA pair-HMM accelerator on AWS F1](https://arodes.hes-so.ch/record/5022/files/author%20version.pdf?version=1)
- [A 28nm Fully Integrated End-to-End Genome Analysis Accelerator for Next-Generation Sequencing](https://scholar.nycu.edu.tw/en/publications/a-28nm-fully-integrated-end-to-end-genome-analysis-accelerator-fo/) — IEEE TBCAS, 2025
- [Genesis: Accelerating sequence analysis (IEEE Micro / Top Picks 2021)](https://users.cs.duke.edu/~lkw34/papers/ham-genesis-toppick2021.pdf)

**Tapeout and DTU**
- [Tiny Tapeout IHP 26b foundry submission](https://github.com/TinyTapeout/tinytapeout-ihp-26b-foundry-submission)
- [Tiny Tapeout IHP 26a](https://tinytapeout.com/chips/ttihp26a/)
- [Tiny Tapeout IHP int8 NPU project (prior art for a MAC tile)](https://github.com/danieltyukov/tt-ihp-int8-npu)
- [Bringing Chip Tapeout Into University Education](https://arxiv.org/pdf/2609.26970)
- [Martin Schoeberl, DTU Compute](https://www.imm.dtu.dk/~masca/)
- [Jens Sparsø, DTU (professor emeritus)](https://www.dtu.dk/person/jens-sparsoe?id=2327)
- [DTU Compute Embedded Systems Engineering — key projects](https://www.compute.dtu.dk/english/research/research-sections/ese/keyprojects)
- [PhD scholarship in Open-Source EDA Tools Development, DTU Compute](https://jobdanmark.dk/job/phd-scholarship-in-open-source-eda-tools-development-dtu-compute)
