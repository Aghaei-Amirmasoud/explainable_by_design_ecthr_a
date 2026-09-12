"""
Main pipeline entry point.

Usage:
    python run_pipeline.py               # full pipeline
    python run_pipeline.py --stage 1     # argument mining only
    python run_pipeline.py --stage 2     # outcome prediction only (needs stage 1 cache)
    python run_pipeline.py --stage eval  # evaluation only (needs both caches)
    python run_pipeline.py --force       # ignore cached outputs, rerun from scratch
"""
import argparse
from pathlib import Path
import config
from utils.helpers import seed_everything


def _run_stage1(force=False):
    from data.data_loader import get_dataset
    from stage1_argument_mining.argument_extractor import LegalBERTArgumentExtractor
    from stage1_argument_mining.sequence_filter import run_stage1, load_stage1, print_stage1_stats

    cache = Path(config.STAGE1_CACHE)
    if not force and cache.exists():
        print("Stage 1 cache found — loading from disk (use --force to rerun).")
        output = load_stage1(cache)
        print_stage1_stats(output)
        return output

    print("=== STAGE 1: Argument Mining (threshold=0.90) ===")
    output = run_stage1(get_dataset(), LegalBERTArgumentExtractor(), save_path=cache)
    print_stage1_stats(output)
    return output


def _run_stage2(stage1_output):
    from stage2_outcome_prediction.embedder import PremiseEmbedder
    from stage2_outcome_prediction.embedder_hybrid import HybridPremiseEmbedder
    from stage2_outcome_prediction.classifier import train_classifier, save_classifier, quick_eval, tune_thresholds
    from stage2_outcome_prediction.classifier import predict_with_thresholds

    print("=== STAGE 2: Outcome Prediction ===")

    # Baseline embedder (full text)
    embedder_base = PremiseEmbedder(use_full_text=True)
    X_train_base, y_train = embedder_base.prepare_split(stage1_output["train"])
    X_val_base,   y_val   = embedder_base.prepare_split(stage1_output["val"])
    X_test_base,  y_test  = embedder_base.prepare_split(stage1_output["test"])
    clf_base = train_classifier(X_train_base, y_train)
    thr_base = tune_thresholds(clf_base, X_val_base, y_val)

    # Premise-only embedder
    embedder_prem = PremiseEmbedder(use_full_text=False)
    X_train_prem, _ = embedder_prem.prepare_split(stage1_output["train"])
    X_val_prem,   _ = embedder_prem.prepare_split(stage1_output["val"])
    X_test_prem,  _ = embedder_prem.prepare_split(stage1_output["test"])
    clf_prem = train_classifier(X_train_prem, y_train)
    thr_prem = tune_thresholds(clf_prem, X_val_prem, y_val)

    # Hybrid embedder
    embedder_hybrid = HybridPremiseEmbedder()
    X_train_hyb, _ = embedder_hybrid.prepare_split(stage1_output["train"])
    X_val_hyb,   _ = embedder_hybrid.prepare_split(stage1_output["val"])
    X_test_hyb,  _ = embedder_hybrid.prepare_split(stage1_output["test"])
    clf_hyb = train_classifier(X_train_hyb, y_train)
    thr_hyb = tune_thresholds(clf_hyb, X_val_hyb, y_val)

    save_classifier(clf_hyb)  # save hybrid as primary

    return {
        "y_test": y_test,
        "baseline": (clf_base, embedder_base, X_test_base, thr_base),
        "premise":  (clf_prem, embedder_prem, X_test_prem, thr_prem),
        "hybrid":   (clf_hyb,  embedder_hybrid, X_test_hyb, thr_hyb),
    }


def _run_evaluation(stage1_output, stage2):
    from stage2_outcome_prediction.classifier import predict_with_thresholds
    from evaluation.metrics import compute_metrics, per_article_f1
    from evaluation.premise_count_analysis import (
        group_cases_by_premise_count, plot_premise_count_analysis, print_premise_count_table)
    from data.data_loader import ARTICLE_NAMES

    y_test = stage2["y_test"]
    results = {}
    for name, (clf, embedder, X_test, thr) in [
        ("Baseline", stage2["baseline"]),
        ("Premise",  stage2["premise"]),
        ("Hybrid",   stage2["hybrid"]),
    ]:
        y_pred = predict_with_thresholds(clf, X_test, thr)
        results[name] = {"metrics": compute_metrics(y_test, y_pred), "y_pred": y_pred}

    print("\n" + "="*70)
    print("RESULTS: Baseline vs Premise vs Hybrid")
    print("="*70)
    print(f"{'Metric':<22} {'Baseline':>15} {'Premise':>15} {'Hybrid':>15}")
    print("="*70)
    for metric in ['macro_f1', 'micro_f1', 'macro_precision', 'macro_recall']:
        row = f"{metric:<22}"
        for name in ("Baseline", "Premise", "Hybrid"):
            row += f" {results[name]['metrics'][metric]:>15.4f}"
        print(row)
    print("="*70)

    print("\n" + "="*80)
    print("PER-ARTICLE F1")
    print("="*80)
    pa = {name: {r['article']: r['f1'] for r in per_article_f1(y_test, results[name]['y_pred'])}
          for name in ("Baseline", "Premise", "Hybrid")}
    print(f"{'Article':<16} {'Baseline':>15} {'Premise':>15} {'Hybrid':>15}")
    print("="*80)
    for art in ARTICLE_NAMES:
        print(f"{art:<16} {pa['Baseline'].get(art,0):>15.4f} "
              f"{pa['Premise'].get(art,0):>15.4f} {pa['Hybrid'].get(art,0):>15.4f}")
    print("="*80)

    preds_dict = {n: results[n]['y_pred'] for n in ("Baseline", "Premise", "Hybrid")}
    pc = group_cases_by_premise_count(stage1_output['test'], preds_dict)
    print("\nPREMISE COUNT ANALYSIS")
    print_premise_count_table(pc)
    plot_premise_count_analysis(
        pc, title="Performance vs Premise Count",
        output_path=config.OUTPUT_DIR / "premise_count_analysis.png")


def _require(*caches):
    for path, hint in caches:
        if not Path(path).exists():
            print(f"ERROR: {path} not found.  {hint}")
            raise SystemExit(1)


def main():
    seed_everything()

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["1", "2", "eval"], default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.stage == "1":
        _run_stage1(force=args.force)

    elif args.stage == "2":
        _require((config.STAGE1_CACHE, "Run: python run_pipeline.py --stage 1"))
        from stage1_argument_mining.sequence_filter import load_stage1
        _run_stage2(load_stage1(Path(config.STAGE1_CACHE)))

    elif args.stage == "eval":
        _require(
            (config.STAGE1_CACHE, "Run: python run_pipeline.py --stage 1"),
            (config.MODEL_CACHE,  "Run: python run_pipeline.py --stage 2"),
        )
        from stage1_argument_mining.sequence_filter import load_stage1
        stage1_output = load_stage1(Path(config.STAGE1_CACHE))
        stage2 = _run_stage2(stage1_output)
        _run_evaluation(stage1_output, stage2)

    else:
        stage1_output = _run_stage1(force=args.force)
        stage2 = _run_stage2(stage1_output)
        _run_evaluation(stage1_output, stage2)

    print(f"Done. Outputs in: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
