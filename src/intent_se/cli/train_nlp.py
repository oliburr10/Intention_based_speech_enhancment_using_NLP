"""Train and evaluate the NLP pipeline.

1. Load the dataset and split it 70 / 15 / 15, stratified.
2. Embed every sentence with ``all-mpnet-base-v2``.
3. Five-fold stratified CV on the development pool (train + val).
4. Fit each candidate on train, score on validation *and* test, and pick the
   classifier with the smallest validation-to-test drop among the CV leaders.
5. Fit the Ridge severity scorer, selecting alpha by CV.
6. Write models, tables and figures to disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intent_se.config import CLASS_ORDER, NLPConfig
from intent_se.nlp import classifier as clf_mod
from intent_se.nlp import evaluate as eval_mod
from intent_se.nlp.dataset import class_distribution, load_dataset, split_dataset
from intent_se.nlp.embeddings import SentenceEmbedder
from intent_se.nlp.severity import SeverityScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_nlp",
        description="Train the intent classifier and severity scorer.",
    )
    parser.add_argument("--data", type=Path, default=None,
                        help="Dataset CSV (default: data/complaints_v4.csv).")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"),
                        help="Where to write models, tables and figures.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/embeddings"),
                        help="Embedding cache directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation (useful on headless machines).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = NLPConfig(random_state=args.seed)
    out = args.output_dir
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    # 1. Data
    df = load_dataset(args.data)
    print(f"Loaded {len(df)} sentences\n")
    print(class_distribution(df).to_string(), "\n")

    split = split_dataset(df, cfg)
    print(split.summary().to_string(), "\n")

    # 2. Embeddings 
    embedder = SentenceEmbedder(cfg, cache_dir=args.cache_dir)
    print(f"Embedding with {cfg.embedding_model} ...")

    dev = split.dev
    x_dev = embedder.encode(dev["sentence"].tolist(), show_progress=True)
    x_train = embedder.encode(split.train["sentence"].tolist(), show_progress=True)
    x_val = embedder.encode(split.val["sentence"].tolist(), show_progress=True)
    x_test = embedder.encode(split.test["sentence"].tolist(), show_progress=True)
    print(f"  development pool: {x_dev.shape}, test: {x_test.shape}\n")

    y_dev = dev["label_id"].to_numpy()
    y_train = split.train["label_id"].to_numpy()
    y_val = split.val["label_id"].to_numpy()
    y_test = split.test["label_id"].to_numpy()

    # 3. Cross-validation
    print(f"{cfg.cv_folds}-fold stratified CV on {len(x_dev)} sentences (macro-F1):")
    cv_results = clf_mod.cross_validate_all(x_dev, y_dev, cfg)
    for result in cv_results:
        print("  ", result)
    cv_table = clf_mod.cv_table(cv_results)
    cv_table.to_csv(out / "tables" / "cross_validation.csv")
    print()

    # 4. Generalisation: val vs test
    print("Fitting candidates on train, scoring on validation and test:")
    rows = []
    fitted = {}
    for result in cv_results:
        model = clf_mod.build_classifiers(cfg)[result.name]
        model.fit(x_train, y_train)
        fitted[result.name] = model
        val = clf_mod.evaluate_on(model, x_val, y_val)
        test = clf_mod.evaluate_on(model, x_test, y_test)
        rows.append({
            "classifier": result.name,
            "cv_f1": round(result.mean_f1, 4),
            "val_f1": round(val["macro_f1"], 4),
            "test_f1": round(test["macro_f1"], 4),
            "val_accuracy": round(val["accuracy"], 4),
            "test_accuracy": round(test["accuracy"], 4),
        })

    gen = eval_mod.generalisation_table(rows)
    print(gen.to_string(), "\n")
    gen.to_csv(out / "tables" / "generalisation.csv")

    # Select among the CV leaders (within one std of the best) by smallest
    # validation-to-test drop -- generalisation, not raw accuracy.
    best_cv = cv_results[0].mean_f1
    contenders = [r.name for r in cv_results if r.mean_f1 >= best_cv - r.std_f1]
    selected = gen.loc[gen.index.isin(contenders), "val_to_test_drop"].idxmin()
    print(f"CV leaders: {contenders}")
    print(f"Selected: {selected} "
          f"(val->test drop {gen.loc[selected, 'val_to_test_drop']:.4f}, "
          f"test macro-F1 {gen.loc[selected, 'test_f1']:.4f})\n")

    # 5. Final classifier on the full development pool
    final = clf_mod.build_classifiers(cfg)[selected]
    intent = clf_mod.IntentClassifier(final).fit(x_dev, y_dev)
    y_pred = intent.predict(x_test)

    print("Test-set classification report:")
    print(eval_mod.report(y_test, y_pred))

    cm = eval_mod.confusion_frame(y_test, y_pred)
    cm.to_csv(out / "tables" / "confusion_matrix.csv")
    print("Confusion matrix (rows = true, cols = predicted):")
    print(cm.to_string(), "\n")

    intent.save(out / "models" / "intent_classifier.joblib")

    # 6. Severity scorer
    complaints_dev = dev["severity"].to_numpy()
    scorer = SeverityScorer(config=cfg)
    search = scorer.select_alpha(x_dev, complaints_dev)
    print("Severity scorer -- alpha selection (CV MAE):")
    print(search.table.to_string())
    print(f"  best: {search}\n")
    search.table.to_csv(out / "tables" / "severity_alpha.csv")

    scorer.fit(x_dev, complaints_dev)
    severity_metrics = scorer.evaluate(x_test, split.test["severity"].to_numpy())
    print(f"Severity on test: MAE {severity_metrics['mae']:.4f}, "
          f"R2 {severity_metrics['r2']:.4f}")

    per_class = scorer.evaluate_per_class(
        x_test, split.test["severity"].to_numpy(), split.test["label"].tolist()
    )
    print("\nSeverity MAE by class:")
    print(per_class.to_string(), "\n")
    per_class.to_csv(out / "tables" / "severity_per_class.csv")

    scorer.save(out / "models" / "severity_scorer.joblib")

    # 7. Figures
    if not args.no_figures:
        print("Writing figures ...")
        eval_mod.plot_confusion_matrix(
            y_test, y_pred, out / "figures" / "confusion_matrix.png"
        )
        eval_mod.plot_tsne(
            x_test, y_test, out / "figures" / "tsne_embeddings.png", cfg
        )
        eval_mod.plot_severity_scatter(
            split.test["severity"].to_numpy(),
            scorer.predict(x_test),
            split.test["label"].tolist(),
            out / "figures" / "severity_scatter.png",
        )

    # 8. Summary
    summary = {
        "n_sentences": int(len(df)),
        "classes": list(CLASS_ORDER),
        "embedding_model": cfg.embedding_model,
        "selected_classifier": selected,
        "cv_macro_f1": float(cv_table.loc[selected, "mean_macro_f1"]),
        "test_macro_f1": float(gen.loc[selected, "test_f1"]),
        "test_accuracy": float(gen.loc[selected, "test_accuracy"]),
        "val_to_test_drop": float(gen.loc[selected, "val_to_test_drop"]),
        "severity_alpha": float(scorer.alpha),
        "severity_cv_mae": float(search.best_mae),
        "severity_test_mae": float(severity_metrics["mae"]),
        "seed": args.seed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nArtifacts written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
