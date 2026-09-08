# Notebooks

The original exploratory work for this project was done in Jupyter notebooks.
The production code in `src/intent_se/` is the organised form of that work.

Drop the original notebooks here to keep them alongside the package. Suggested
naming:

```
notebooks/
├── 01_audio_pipeline_exploration.ipynb
├── 02_dataset_construction.ipynb
├── 03_classifier_comparison.ipynb
└── 04_severity_scorer.ipynb
```

To keep the diffs readable, strip outputs before committing:

```bash
pip install nbstripout
nbstripout --install          # registers a git filter for this repo
```

Notebooks are for exploration; anything that needs to run reproducibly belongs
in `src/intent_se/` with a test.
