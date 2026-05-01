import random
from pathlib import Path
import numpy as np
from itertools import chain
from stage2_outcome_prediction.traceback import print_justified_verdict, trace_batch
import config


def _categorise(verdict):
    predicted = set(verdict["predicted_articles"])
    true      = set(verdict["true_articles"])
    if predicted & true:        return "TP"
    if not predicted and true:  return "FN"
    if predicted:               return "FP"
    return "TN"


def select_review_cases(verdicts, n=config.QUALITATIVE_N, seed=config.RANDOM_SEED):
    buckets = {"TP": [], "TN": [], "FP": [], "FN": []}
    for i, verdict in enumerate(verdicts):
        cat = _categorise(verdict)
        buckets[cat].append({"verdict": verdict, "category": cat, "idx": i})

    rng        = random.Random(seed)
    per_bucket = max(1, n // 4)
    selected   = []
    for cases in buckets.values():
        rng.shuffle(cases)
        selected.extend(cases[:per_bucket])

    selected_set = set(id(c) for c in selected)
    remaining = [c for c in chain.from_iterable(buckets.values()) if id(c) not in selected_set]
    rng.shuffle(remaining)
    selected.extend(remaining[:max(0, n - len(selected))])
    return selected


def generate_markdown_report(selected_cases, output_path=config.OUTPUT_DIR / "qualitative_review.md", top_k=3):
    def to_md(item):
        v, cat = item["verdict"], item["category"]
        lines = [
            f"## Case {v['case_id']}  [{cat}]", "",
            f"**True articles**      : {', '.join(v['true_articles']) or 'No violation'}",
            f"**Predicted articles** : {', '.join(v['predicted_articles']) or 'No violation'}", "",
        ]
        if v["justifications"]:
            lines.append("### Justifications")
            for j in v["justifications"]:
                lines.append(f"**{j['article']}**")
                for rank, p in enumerate(j["supporting_premises"][:top_k], 1):
                    lines.append(f"{rank}. *(attribution={p['attribution']:.4f})* {p['sentence']}")
                lines.append("")
        else:
            lines.append("*No violations predicted.*")
        lines.append("---")
        return "\n".join(lines)

    md = "\n".join([
        "# Qualitative Review: Legal Outcome Prediction", "",
        f"Total cases reviewed: {len(selected_cases)}", "",
        *[to_md(item) for item in selected_cases],
    ])
    Path(output_path).write_text(md, encoding="utf-8")
    return md


def run_qualitative_review(test_cases, X_test, clf, embedder,
                           n=config.QUALITATIVE_N, top_k=3, save_report=True):
    verdicts = trace_batch(test_cases, X_test, clf, embedder, top_k=top_k)
    selected = select_review_cases(verdicts, n=n)

    for item in selected:
        print(f"\n{'=' * 70}\n  CATEGORY: {item['category']}")
        print_justified_verdict(item["verdict"])

    if save_report:
        generate_markdown_report(selected, top_k=top_k)
    return selected
