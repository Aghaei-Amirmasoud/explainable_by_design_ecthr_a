import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, hamming_loss
from data.data_loader import ARTICLE_NAMES
import config

EVAL_ARTICLE_NAMES = ARTICLE_NAMES + ["No Violation"]


def _append_no_violation(y):
    """Append a 'no violation' column: 1 if all other labels are 0, else 0."""
    y = np.asarray(y)
    no_viol = (y.sum(axis=1) == 0).astype(int).reshape(-1, 1)
    return np.hstack([y, no_viol])


def compute_metrics(y_true, y_pred):
    y_true_ext = _append_no_violation(y_true)
    y_pred_ext = _append_no_violation(y_pred)
    return {
        "macro_f1":        f1_score(y_true_ext, y_pred_ext, average="macro",  zero_division=0),
        "micro_f1":        f1_score(y_true_ext, y_pred_ext, average="micro",  zero_division=0),
        "macro_precision": precision_score(y_true_ext, y_pred_ext, average="macro", zero_division=0),
        "macro_recall":    recall_score(y_true_ext, y_pred_ext,    average="macro", zero_division=0),
        "micro_precision": precision_score(y_true_ext, y_pred_ext, average="micro", zero_division=0),
        "micro_recall":    recall_score(y_true_ext, y_pred_ext,    average="micro", zero_division=0),
        "hamming_loss":    hamming_loss(y_true_ext, y_pred_ext),
    }


def print_metrics(metrics, title="Metrics"):
    print(f"\n{'─' * 50}\n  {title}\n{'─' * 50}")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v:.4f}")
    print(f"{'─' * 50}")


def print_classification_report(y_true, y_pred, label_names=None):
    y_true_ext = _append_no_violation(y_true)
    y_pred_ext = _append_no_violation(y_pred)
    print(classification_report(y_true_ext, y_pred_ext,
                                target_names=label_names or EVAL_ARTICLE_NAMES,
                                zero_division=0))


def compare_classifiers(y_true, y_pred_baseline, y_pred_pipeline):
    m_base = compute_metrics(y_true, y_pred_baseline)
    m_pipe = compute_metrics(y_true, y_pred_pipeline)
    delta  = {k: m_pipe[k] - m_base[k] for k in m_base}
    print_metrics(m_base, "BASELINE (raw text)")
    print_metrics(m_pipe, "PIPELINE (extracted premises)")
    print_metrics(delta,  "Δ (pipeline − baseline)")
    return {"baseline": m_base, "pipeline": m_pipe, "delta": delta}


def per_article_f1(y_true, y_pred):
    y_true_ext = _append_no_violation(y_true)
    y_pred_ext = _append_no_violation(y_pred)
    scores  = f1_score(y_true_ext, y_pred_ext, average=None, zero_division=0)
    support = y_true_ext.sum(axis=0).astype(int)
    results = [{"article": EVAL_ARTICLE_NAMES[i], "f1": round(float(scores[i]), 4), "support": int(support[i])}
               for i in range(len(scores))]
    results.sort(key=lambda x: x["f1"], reverse=True)
    return results


def print_per_article_f1(results, top_n=15):
    print(f"\n{'─' * 45}\n  Per-Article F1 (top {top_n})\n{'─' * 45}")
    print(f"  {'Article':<12}  {'F1':>6}  {'Support':>8}")
    print(f"{'─' * 45}")
    for row in results[:top_n]:
        print(f"  {row['article']:<12}  {row['f1']:>6.4f}  {row['support']:>8}")
    print(f"{'─' * 45}")


def train_baseline_classifier(stage1_train, stage1_test, embedder, stage1_val=None):
    """Train baseline classifier using full text with paragraph-level embedding.

    Embeds each paragraph individually (avoiding truncation), then pools them
    using the same mechanism as premise-only for fair comparison.

    Pass `stage1_val` to tune per-article thresholds on the validation split,
    matching what the premise and hybrid classifiers do. Without it the baseline
    is evaluated at the default decision boundary (0) while its competitors are
    threshold-tuned, which understates the baseline.
    """
    from stage2_outcome_prediction.classifier import (
        train_classifier, predict, tune_thresholds, predict_with_thresholds)

    def paragraphs_as_premises(cases):
        """Convert paragraphs to premise format for fair paragraph-level embedding.

        This ensures baseline doesn't suffer from truncation bias:
        - Each paragraph embedded separately (no 384-token truncation)
        - Pooled using same concatenated pooling as premise-only
        - Fair comparison across all approaches
        """
        baseline_cases = []
        for case in cases:
            # Treat each paragraph as a "premise" so they get embedded individually
            fake_premises = [
                {
                    "sentence": para,
                    "confidence": 1.0,  # Equal weight for all paragraphs
                    "paragraph_id": i,
                    "sentence_id": 0
                }
                for i, para in enumerate(case.get("paragraphs", []))
            ]
            baseline_cases.append({
                **case,
                "premises": fake_premises
            })
        return baseline_cases

    # Use same prepare_split mechanism as premise-only (fair comparison)
    X_train, y_train = embedder.prepare_split(paragraphs_as_premises(stage1_train))
    X_test, y_test = embedder.prepare_split(paragraphs_as_premises(stage1_test))

    clf = train_classifier(X_train, y_train)

    if stage1_val is not None:
        X_val, y_val = embedder.prepare_split(paragraphs_as_premises(stage1_val))
        thresholds = tune_thresholds(clf, X_val, y_val)
        y_pred = predict_with_thresholds(clf, X_test, thresholds)
    else:
        y_pred = predict(clf, X_test)
    return y_pred, clf
