import numpy as np

from stage2_outcome_prediction.traceback import (
    attribute_premises_svm, embed_individual_premises)


def _subset_case(case, keep_idx):
    prem = case.get("premises", [])
    return {**case, "premises": [prem[i] for i in sorted(keep_idx)]}


def _margins(clf, X, label_idx):
    return clf.estimators_[label_idx].decision_function(X)


def build_rationales(cases, X, clf, embedder, k=3, seed=0, max_cases=None):
    rng = np.random.default_rng(seed)
    jobs = []
    n = len(cases) if max_cases is None else min(len(cases), max_cases)

    for ci in range(n):
        case = cases[ci]
        prem = case.get("premises", [])
        if len(prem) <= k:
            continue
        y_pred = np.array([int(_margins(clf, X[ci:ci + 1], j)[0] > 0)
                           for j in range(len(clf.estimators_))])
        if not y_pred.any():
            continue
        prem_embs = embed_individual_premises(prem, embedder)
        if len(prem_embs) == 0:
            continue
        for j in np.flatnonzero(y_pred):
            ranked = attribute_premises_svm(X[ci], clf.estimators_[j], prem_embs, top_k=k)
            top = [i for i, _ in ranked]
            if not top:
                continue
            rand = list(rng.choice(len(prem), size=len(top), replace=False))
            jobs.append({"ci": ci, "label": int(j), "top": top, "rand": rand,
                         "n_prem": len(prem)})
    return jobs


def evaluate(cases, X, clf, embedder, k=3, seed=0, max_cases=None, verbose=True):
    jobs = build_rationales(cases, X, clf, embedder, k, seed, max_cases)
    if not jobs:
        raise ValueError("no eligible (case, article) pairs")
    if verbose:
        print(f"  {len(jobs)} (case, predicted-article) pairs, k={k}")

    # four ablated variants per job, embedded in one batch each
    variants = {
        "minus_top":  [_subset_case(cases[j["ci"]],
                                    set(range(j["n_prem"])) - set(j["top"])) for j in jobs],
        "only_top":   [_subset_case(cases[j["ci"]], j["top"])  for j in jobs],
        "minus_rand": [_subset_case(cases[j["ci"]],
                                    set(range(j["n_prem"])) - set(j["rand"])) for j in jobs],
        "only_rand":  [_subset_case(cases[j["ci"]], j["rand"]) for j in jobs],
    }

    full_margin = np.array([_margins(clf, X[j["ci"]:j["ci"] + 1], j["label"])[0] for j in jobs])
    out = {}
    for name, vcases in variants.items():
        if verbose:
            print(f"  embedding {name} ...", flush=True)
        Xv, _ = embedder.prepare_split(vcases)
        out[name] = np.array([_margins(clf, Xv[i:i + 1], j["label"])[0]
                              for i, j in enumerate(jobs)])

    res = {
        "k": k, "n_pairs": len(jobs),
        "comprehensiveness_top":  float((full_margin - out["minus_top"]).mean()),
        "comprehensiveness_rand": float((full_margin - out["minus_rand"]).mean()),
        "sufficiency_top":        float((full_margin - out["only_top"]).mean()),
        "sufficiency_rand":       float((full_margin - out["only_rand"]).mean()),
        "flip_rate_top":          float((out["minus_top"] <= 0).mean()),
        "flip_rate_rand":         float((out["minus_rand"] <= 0).mean()),
        "retained_top":           float((out["only_top"] > 0).mean()),
        "retained_rand":          float((out["only_rand"] > 0).mean()),
    }
    res["comprehensiveness_gain"] = res["comprehensiveness_top"] - res["comprehensiveness_rand"]
    res["sufficiency_gain"]       = res["sufficiency_rand"] - res["sufficiency_top"]
    return res


def print_report(r):
    print("=" * 72)
    print(f"  FAITHFULNESS OF PREMISE ATTRIBUTIONS  (k={r['k']}, n={r['n_pairs']} pairs)")
    print("=" * 72)
    print(f"  {'metric':<34} {'top-k':>10} {'random-k':>10} {'gap':>10}")
    print("  " + "-" * 66)
    print(f"  {'comprehensiveness (higher better)':<34} {r['comprehensiveness_top']:>10.4f}"
          f" {r['comprehensiveness_rand']:>10.4f} {r['comprehensiveness_gain']:>+10.4f}")
    print(f"  {'sufficiency (lower better)':<34} {r['sufficiency_top']:>10.4f}"
          f" {r['sufficiency_rand']:>10.4f} {r['sufficiency_gain']:>+10.4f}")
    print(f"  {'prediction flips when removed':<34} {r['flip_rate_top']:>9.1%}"
          f" {r['flip_rate_rand']:>10.1%} {r['flip_rate_top'] - r['flip_rate_rand']:>+10.1%}")
    print(f"  {'prediction held by rationale alone':<34} {r['retained_top']:>9.1%}"
          f" {r['retained_rand']:>10.1%} {r['retained_top'] - r['retained_rand']:>+10.1%}")
    print("  " + "-" * 66)

    print("\n  VERDICT")
    if r["comprehensiveness_gain"] > 0 and r["sufficiency_gain"] > 0:
        print("    The cited premises matter MORE than random ones on both axes:")
        print("    removing them hurts the prediction more, and they carry it better")
        print("    on their own. The attributions are faithful.")
    elif r["comprehensiveness_gain"] > 0:
        print("    Removing the cited premises hurts more than removing random ones,")
        print("    but they are not more self-sufficient. Partially faithful -")
        print("    report both numbers.")
    elif r["sufficiency_gain"] > 0:
        print("    The cited premises reconstruct the prediction better than random")
        print("    ones, but removing them is no more damaging. Partially faithful.")
    else:
        print("    The cited premises behave no differently from randomly chosen")
        print("    ones. The explanations are NOT faithful to the classifier -")
        print("    say so; it is a real finding about attribution-by-projection.")
