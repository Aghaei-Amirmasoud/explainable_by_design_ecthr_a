"""
Main pipeline entry point.

Usage:
    python run_pipeline.py               # full pipeline
    python run_pipeline.py --stage 1     # argument mining only
    python run_pipeline.py --stage 2     # outcome prediction only (needs stage 1 cache)
    python run_pipeline.py --stage eval  # evaluation only (needs both caches)
    python run_pipeline.py --no-eval     # stages 1+2, skip evaluation
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

    if not force and Path(config.STAGE1_CACHE).exists():
        print("Stage 1 cache found — loading from disk (use --force to rerun).")
        output = load_stage1()
        print_stage1_stats(output)
        return output

    print("=== STAGE 1: Argument Mining ===")
    output = run_stage1(get_dataset(), LegalBERTArgumentExtractor())
    print_stage1_stats(output)
    return output


def _run_stage2(stage1_output):
    from stage2_outcome_prediction.classifier import train_classifier, save_classifier, quick_eval

    print("=== STAGE 2: Outcome Prediction ===")

    # Choose embedder based on config flag
    if config.USE_HYBRID_EMBEDDER:
        from stage2_outcome_prediction.embedder_hybrid import HybridPremiseEmbedder
        embedder = HybridPremiseEmbedder()
        print("  Using HYBRID embedder (full paragraphs + premise features)")
    else:
        from stage2_outcome_prediction.embedder import PremiseEmbedder
        embedder = PremiseEmbedder()
        print("  Using PREMISE-ONLY embedder (current approach)")

    X_train, y_train = embedder.prepare_split(stage1_output["train"])
    X_test,  y_test  = embedder.prepare_split(stage1_output["test"])
    clf = train_classifier(X_train, y_train)

    val_cases = stage1_output.get("val", [])
    if val_cases:
        quick_eval(clf, *embedder.prepare_split(val_cases))

    save_classifier(clf)
    return clf, embedder, X_test, y_test


def _run_evaluation(stage1_output, clf, embedder, X_test, y_test):
    from stage2_outcome_prediction.classifier import predict, predict_with_thresholds, tune_thresholds
    from evaluation.metrics import (compute_metrics, per_article_f1, train_baseline_classifier)
    from evaluation.qualitative_review import run_qualitative_review
    from evaluation.premise_count_analysis import (
        group_cases_by_premise_count, plot_premise_count_analysis, print_premise_count_table)
    from data.data_loader import ARTICLE_NAMES

    print("="*70)
    print("THREE-WAY EVALUATION: Baseline vs Premise vs Hybrid")
    print("="*70)

    # 1. Baseline (raw text, no premise extraction)
    print("\n[1/3] Training baseline (raw text)...")
    y_pred_baseline, clf_baseline = train_baseline_classifier(
        stage1_output['train'], stage1_output['test'], embedder)

    # 2. Premise (current approach)
    print("\n[2/3] Evaluating premise classifier...")
    X_val, y_val = embedder.prepare_split(stage1_output['val'])
    thresholds_premise = tune_thresholds(clf, X_val, y_val)
    print("  Tuned thresholds:")
    for name, t in zip(ARTICLE_NAMES, thresholds_premise):
        print(f"    {name:<12} threshold={t:+.4f}")
    y_pred_premise = predict_with_thresholds(clf, X_test, thresholds_premise)

    # 3. Hybrid (if enabled)
    if config.USE_HYBRID_EMBEDDER:
        print("\n[3/3] Evaluating hybrid classifier...")
        from stage2_outcome_prediction.embedder_hybrid import HybridPremiseEmbedder
        embedder_hybrid = HybridPremiseEmbedder()
        X_train_hybrid, y_train_hybrid = embedder_hybrid.prepare_split(stage1_output['train'])
        X_val_hybrid, y_val_hybrid = embedder_hybrid.prepare_split(stage1_output['val'])
        X_test_hybrid, y_test_hybrid = embedder_hybrid.prepare_split(stage1_output['test'])

        from stage2_outcome_prediction.classifier import train_classifier
        clf_hybrid = train_classifier(X_train_hybrid, y_train_hybrid)
        thresholds_hybrid = tune_thresholds(clf_hybrid, X_val_hybrid, y_val_hybrid)
        y_pred_hybrid = predict_with_thresholds(clf_hybrid, X_test_hybrid, thresholds_hybrid)

        # Three-way comparison
        metrics_baseline = compute_metrics(y_test, y_pred_baseline)
        metrics_premise = compute_metrics(y_test, y_pred_premise)
        metrics_hybrid = compute_metrics(y_test_hybrid, y_pred_hybrid)

        print("\n" + "="*70)
        print("OVERALL METRICS")
        print("="*70)
        print(f"{'Metric':<22} {'Baseline':>15} {'Premise':>15} {'Hybrid':>15}")
        print("="*70)
        for metric in ['macro_f1', 'micro_f1', 'macro_precision', 'macro_recall',
                       'micro_precision', 'micro_recall']:
            b = metrics_baseline[metric]
            p = metrics_premise[metric]
            h = metrics_hybrid[metric]
            print(f"{metric:<22} {b:>15.4f} {p:>15.4f} {h:>15.4f}")
        print("="*70)

        # Per-article comparison
        pa_f1_baseline = per_article_f1(y_test, y_pred_baseline)
        pa_f1_premise = per_article_f1(y_test, y_pred_premise)
        pa_f1_hybrid = per_article_f1(y_test_hybrid, y_pred_hybrid)

        print("\n" + "="*80)
        print("PER-ARTICLE F1 SCORES")
        print("="*80)
        print(f"{'Article':<16} {'Baseline':>15} {'Premise':>15} {'Hybrid':>15} {'Support':>10}")
        print("="*80)
        for r in pa_f1_baseline:
            article = r['article']
            b = {r['article']: r['f1'] for r in pa_f1_baseline}.get(article, 0.0)
            p = {r['article']: r['f1'] for r in pa_f1_premise}.get(article, 0.0)
            h = {r['article']: r['f1'] for r in pa_f1_hybrid}.get(article, 0.0)
            support = r['support']
            print(f"{article:<16} {b:>15.4f} {p:>15.4f} {h:>15.4f} {support:>10}")
        print("="*80)

        best = max([('Baseline', metrics_baseline['macro_f1']),
                    ('Premise', metrics_premise['macro_f1']),
                    ('Hybrid', metrics_hybrid['macro_f1'])], key=lambda x: x[1])
        print(f"\nBest approach: {best[0]} (macro F1: {best[1]:.4f})")

        # Premise count analysis
        print("\n" + "="*70)
        print("PREMISE COUNT ANALYSIS")
        print("="*70)
        predictions_classical = {
            'Baseline': y_pred_baseline,
            'Premise': y_pred_premise,
            'Hybrid': y_pred_hybrid
        }
        results_classical = group_cases_by_premise_count(stage1_output['test'], predictions_classical)
        print("\nPerformance by number of extracted premises:")
        print_premise_count_table(results_classical)
        plot_premise_count_analysis(
            results_classical,
            title="Classical ML: Performance vs Premise Count",
            output_path=config.OUTPUT_DIR / "premise_count_analysis_classical.png",
            colors={'Baseline': '#1f77b4', 'Premise': '#ff7f0e', 'Hybrid': '#2ca02c'}
        )
    else:
        # Two-way comparison (baseline vs premise)
        metrics_baseline = compute_metrics(y_test, y_pred_baseline)
        metrics_premise = compute_metrics(y_test, y_pred_premise)

        print("\n" + "="*70)
        print("OVERALL METRICS")
        print("="*70)
        print(f"{'Metric':<22} {'Baseline':>15} {'Premise':>15}")
        print("="*70)
        for metric in ['macro_f1', 'micro_f1', 'macro_precision', 'macro_recall']:
            b = metrics_baseline[metric]
            p = metrics_premise[metric]
            print(f"{metric:<22} {b:>15.4f} {p:>15.4f}")
        print("="*70)

        pa_f1_results = per_article_f1(y_test, y_pred_premise)
        print("\n" + "="*80)
        print("PER-ARTICLE F1 SCORES (Premise)")
        print("="*80)
        for r in pa_f1_results:
            print(f"{r['article']:<16} {r['f1']:>15.4f} {r['support']:>10}")
        print("="*80)

    # Qualitative review
    print("\n" + "="*70)
    print("QUALITATIVE REVIEW (Premise Classifier)")
    print("="*70)
    run_qualitative_review(stage1_output['test'], X_test, clf, embedder)


def _require(*caches):
    for path, hint in caches:
        if not Path(path).exists():
            print(f"ERROR: {path} not found.  {hint}")
            raise SystemExit(1)


def main():
    seed_everything()

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",   choices=["1", "2", "eval"], default=None)
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    print(f"Config: CLASSIFIER_TYPE={config.CLASSIFIER_TYPE}, USE_HYBRID_EMBEDDER={config.USE_HYBRID_EMBEDDER}")

    if args.stage == "1":
        _run_stage1(force=args.force)

    elif args.stage == "2":
        _require((config.STAGE1_CACHE, "Run: python run_pipeline.py --stage 1"))
        from stage1_argument_mining.sequence_filter import load_stage1
        _run_stage2(load_stage1())

    elif args.stage == "eval":
        _require(
            (config.STAGE1_CACHE, "Run: python run_pipeline.py --stage 1"),
            (config.MODEL_CACHE,  "Run: python run_pipeline.py --stage 2"),
        )
        from stage1_argument_mining.sequence_filter import load_stage1
        from stage2_outcome_prediction.classifier import load_classifier

        # Choose embedder based on config flag (must match training)
        if config.USE_HYBRID_EMBEDDER:
            from stage2_outcome_prediction.embedder_hybrid import HybridPremiseEmbedder
            embedder = HybridPremiseEmbedder()
        else:
            from stage2_outcome_prediction.embedder import PremiseEmbedder
            embedder = PremiseEmbedder()

        stage1_output  = load_stage1()
        X_test, y_test = embedder.prepare_split(stage1_output["test"])
        _run_evaluation(stage1_output, load_classifier(), embedder, X_test, y_test)

    else:
        stage1_output              = _run_stage1(force=args.force)
        clf, embedder, X_test, y_test = _run_stage2(stage1_output)
        if not args.no_eval:
            _run_evaluation(stage1_output, clf, embedder, X_test, y_test)

    print(f"Done. Outputs in: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
