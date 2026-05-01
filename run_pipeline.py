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
    from stage2_outcome_prediction.embedder import PremiseEmbedder
    from stage2_outcome_prediction.classifier import train_classifier, save_classifier, quick_eval

    print("=== STAGE 2: Outcome Prediction ===")
    embedder = PremiseEmbedder()

    X_train, y_train = embedder.prepare_split(stage1_output["train"])
    X_test,  y_test  = embedder.prepare_split(stage1_output["test"])
    clf = train_classifier(X_train, y_train)

    val_cases = stage1_output.get("val", [])
    if val_cases:
        quick_eval(clf, *embedder.prepare_split(val_cases))

    save_classifier(clf)
    return clf, embedder, X_test, y_test


def _run_evaluation(stage1_output, clf, embedder, X_test, y_test):
    from stage2_outcome_prediction.classifier import predict
    from evaluation.metrics import (compare_classifiers, per_article_f1,
                                    print_per_article_f1, train_baseline_classifier)
    from evaluation.qualitative_review import run_qualitative_review
    from visualize import generate_all_plots

    print("=== EVALUATION ===")
    y_pred_pipeline = predict(clf, X_test)
    y_pred_baseline, _ = train_baseline_classifier(stage1_output["train"], stage1_output["test"], embedder)

    comparison    = compare_classifiers(y_test, y_pred_baseline, y_pred_pipeline)
    pa_f1_results = per_article_f1(y_test, y_pred_pipeline)
    print_per_article_f1(pa_f1_results)

    run_qualitative_review(stage1_output["test"], X_test, clf, embedder)
    generate_all_plots(comparison=comparison, pa_f1_results=pa_f1_results,
                       stage1_output=stage1_output, clf=clf, X_test=X_test)
    return comparison


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

    print(f"Config: CLASSIFIER_TYPE={config.CLASSIFIER_TYPE}")

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
        from stage2_outcome_prediction.embedder import PremiseEmbedder
        stage1_output  = load_stage1()
        embedder       = PremiseEmbedder()
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
