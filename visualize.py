import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import config
from data.data_loader import ARTICLE_NAMES

plt.rcParams.update({
    "figure.dpi": 150, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
})

PALETTE = {"baseline": "#9E9E9E", "pipeline": "#4DD0E1"}


def plot_f1_comparison(comparison, save_path=str(config.OUTPUT_DIR / "f1_comparison.png")):
    metrics  = ["macro_f1", "micro_f1"]
    baseline = [comparison["baseline"][m] for m in metrics]
    pipeline = [comparison["pipeline"][m] for m in metrics]
    x, w     = np.arange(2), 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_b  = ax.bar(x - w/2, baseline, w, label="Baseline", color=PALETTE["baseline"], alpha=0.85)
    bars_p  = ax.bar(x + w/2, pipeline, w, label="Pipeline", color=PALETTE["pipeline"], alpha=0.85)
    for bar in [*bars_b, *bars_p]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set(ylim=(0, 1.1), ylabel="Score", xlabel="Metric",
           title="Baseline vs. Pipeline", xticks=x, xticklabels=["Macro-F1", "Micro-F1"])
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def plot_per_article_f1(pa_f1_results, save_path=str(config.OUTPUT_DIR / "per_article_f1.png"), top_n=20):
    top      = pa_f1_results[:top_n]
    articles = [r["article"] for r in top]
    f1_vals  = [r["f1"]      for r in top]
    colours  = [PALETTE["pipeline"] if r["support"] > 0 else PALETTE["baseline"] for r in top]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    bars = ax.barh(articles[::-1], f1_vals[::-1], color=colours[::-1], alpha=0.85)
    for bar, val in zip(bars, f1_vals[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.set(xlim=(0, 1.1), xlabel="F1 Score", title=f"Per-Article F1 (top {top_n})")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def plot_premise_confidence(stage1_output, save_path=str(config.OUTPUT_DIR / "premise_confidence.png")):
    confs = [p["confidence"] for cases in stage1_output.values()
             for case in cases for p in case.get("premises", [])]
    if not confs:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(confs, bins=30, color=PALETTE["pipeline"], alpha=0.85, edgecolor="white")
    ax.axvline(config.PREMISE_THRESHOLD, color="red", linestyle="--",
               label=f"Threshold={config.PREMISE_THRESHOLD}")
    ax.set(xlabel="Confidence", ylabel="Count", title="Stage 1 – Premise Confidence")
    ax.legend(); ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def plot_label_distribution(stage1_output, save_path=str(config.OUTPUT_DIR / "label_distribution.png"), split="train"):
    cases = stage1_output.get(split, [])
    if not cases:
        return
    counts = sum(np.array(c["labels_binary"]) for c in cases)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(ARTICLE_NAMES, counts, color=PALETTE["pipeline"], alpha=0.85)
    ax.set(xlabel="ECHR Article", ylabel="Frequency", title=f"Label Distribution – {split}")
    plt.xticks(rotation=60, ha="right", fontsize=7)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def plot_decision_path_depths(clf, X_test, save_path=str(config.OUTPUT_DIR / "decision_path_depths.png")):
    if config.CLASSIFIER_TYPE != "decision_tree":
        return
    from sklearn.tree import DecisionTreeClassifier
    depths = [estimator.decision_path(X_test).getrow(i).nnz
              for estimator in clf.estimators_
              if isinstance(estimator, DecisionTreeClassifier)
              for i in range(X_test.shape[0])]
    if not depths:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(depths, bins=30, color=PALETTE["pipeline"], alpha=0.85, edgecolor="white")
    ax.set(xlabel="Path Length (nodes)", ylabel="Count", title="Stage 2 – Decision Path Depth")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def generate_all_plots(comparison=None, pa_f1_results=None,
                       stage1_output=None, clf=None, X_test=None):
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    if comparison:       plot_f1_comparison(comparison)
    if pa_f1_results:    plot_per_article_f1(pa_f1_results)
    if stage1_output:    plot_premise_confidence(stage1_output); plot_label_distribution(stage1_output)
    if clf is not None and X_test is not None:
        plot_decision_path_depths(clf, X_test)
    print(f"Plots saved to: {config.OUTPUT_DIR}")
