import json
import random
from pathlib import Path

import numpy as np

import config
from stage1_argument_mining.argument_extractor import _split_sentences

CONDITIONS = ("premises", "complement", "random", "lead")
RESULTS_PATH = config.OUTPUT_DIR / "extractor_controls.json"


# Building the variants

def case_sentences(case):
    out = []
    for pid, para in enumerate(case.get("paragraphs", [])):
        for sid, sent in enumerate(_split_sentences(para)):
            out.append((pid, sid, sent))
    return out


def build_variant(case, condition, rng):
    premises = case.get("premises", [])
    if not premises or condition == "premises":
        return premises

    target = sum(len(p["sentence"].split()) for p in premises)
    chosen = {(p.get("paragraph_id", -1), p.get("sentence_id", -1)) for p in premises}
    sents  = case_sentences(case)

    if condition == "complement":
        pool = [s for s in sents if (s[0], s[1]) not in chosen]
        rng.shuffle(pool)
    elif condition == "random":
        pool = list(sents)
        rng.shuffle(pool)
    elif condition == "lead":
        pool = list(sents)                       # already in document order
    else:
        raise ValueError(f"unknown condition: {condition}")

    picked, words = [], 0
    for pid, sid, text in pool:
        if words >= target:
            break
        picked.append({"sentence": text, "confidence": 1.0,
                       "paragraph_id": pid, "sentence_id": sid})
        words += len(text.split())

    # A case whose complement is empty (every sentence was a premise) keeps its
    # premises; otherwise it would silently become a full-text fallback and the
    # conditions would no longer be comparable.
    return picked or premises


def apply_condition(cases, condition, seed=0):
    rng = random.Random(seed)
    return [{**c, "premises": build_variant(c, condition, rng)} for c in cases]


def budget_report(cases, conditions=CONDITIONS, seed=0):
    print(f"  {'condition':<12} {'median words':>13} {'median sents':>13}")
    print("  " + "-" * 40)
    for cond in conditions:
        cs = apply_condition(cases, cond, seed)
        w = [sum(len(p["sentence"].split()) for p in c["premises"])
             for c in cs if c.get("premises")]
        n = [len(c["premises"]) for c in cs if c.get("premises")]
        print(f"  {cond:<12} {np.median(w):>13.0f} {np.median(n):>13.0f}")
    full = np.median([len(" ".join(c.get("paragraphs", [])).split()) for c in cases])
    print(f"  {'(full text)':<12} {full:>13.0f}")


# Running one condition through the SVM pipeline
def run_condition(condition, stage1, embedder, seed=0):
    from stage2_outcome_prediction.classifier import (
        train_classifier, tune_thresholds, predict_with_thresholds)
    from evaluation.metrics import compute_metrics, per_article_f1

    tr = apply_condition(stage1["train"], condition, seed)
    va = apply_condition(stage1["val"],   condition, seed)
    te = apply_condition(stage1["test"],  condition, seed)

    X_tr, y_tr = embedder.prepare_split(tr)
    X_va, y_va = embedder.prepare_split(va)
    X_te, _    = embedder.prepare_split(te)

    clf    = train_classifier(X_tr, y_tr)
    thr    = tune_thresholds(clf, X_va, y_va)
    y_pred = predict_with_thresholds(clf, X_te, thr)

    y_test = np.array([c["labels_binary"] for c in stage1["test"]])
    m = compute_metrics(y_test, y_pred)
    m["per_article"] = per_article_f1(y_test, y_pred)
    m["condition"]   = condition
    return m


def load_results(path=RESULTS_PATH):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def save_results(results, path=RESULTS_PATH):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2, default=float))
    return p


def run_all(stage1, embedder, conditions=CONDITIONS, seed=0, force=False,
            path=RESULTS_PATH):
    results = {} if force else load_results(path)
    for cond in conditions:
        if cond in results:
            r = results[cond]
            print(f"  {cond:<12} macro={r['macro_f1']:.4f}  micro={r['micro_f1']:.4f}   (cached)")
            continue
        print(f"  {cond:<12} embedding + training ...", flush=True)
        results[cond] = run_condition(cond, stage1, embedder, seed)
        save_results(results, path)
        r = results[cond]
        print(f"  {cond:<12} macro={r['macro_f1']:.4f}  micro={r['micro_f1']:.4f}")
    return results


# Reporting

def print_report(results, conditions=CONDITIONS):
    have = [c for c in conditions if c in results]
    print("=" * 62)
    print("EXTRACTOR VALIDATION - matched-budget controls (SVM, deterministic)")
    print("=" * 62)
    print(f"  {'condition':<12} {'Macro F1':>10} {'Micro F1':>10}")
    print("  " + "-" * 34)
    for c in have:
        print(f"  {c:<12} {results[c]['macro_f1']:>10.4f} {results[c]['micro_f1']:>10.4f}")

    if "premises" not in results:
        print("\n  (run the 'premises' condition to get the comparison)")
        return

    print(f"\n  premises minus each control:")
    gaps = {}
    for c in have:
        if c == "premises":
            continue
        gaps[c] = (results["premises"]["macro_f1"] - results[c]["macro_f1"],
                   results["premises"]["micro_f1"] - results[c]["micro_f1"])
        print(f"    vs {c:<12} macro {gaps[c][0]:+.4f}   micro {gaps[c][1]:+.4f}")

    if not gaps:
        return
    print("\n" + "=" * 62)
    print("  VERDICT")
    print("=" * 62)
    worst = min(g[0] for g in gaps.values())
    if worst > 0.03:
        print("  Premises beat every matched-budget control by a clear margin.")
        print("  -> Stage 1 is selecting signal, not just selecting less text.")
        print("     'Our extracted premises are really premises' is now measured,")
        print("     not inferred from the small full-text gap.")
    elif worst > 0.01:
        print("  Premises lead every control, but modestly.")
        print("  -> Some real selection signal; report the gaps honestly and do")
        print("     not lean on premise quality as a headline claim.")
    else:
        weak = [c for c, g in gaps.items() if g[0] <= 0.01]
        print(f"  Premises do NOT clearly beat: {', '.join(weak)}")
        print("  -> At matched budget the extractor's choice barely matters; the")
        print("     0.06 full-text gap is mostly the cost of compression, not")
        print("     evidence of premise quality. This needs saying out loud.")
        if "lead" in weak:
            print("     'lead' in particular suggests position bias, not argument detection.")
