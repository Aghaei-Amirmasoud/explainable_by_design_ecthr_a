import numpy as np
from sklearn.svm import LinearSVC
from data.data_loader import ARTICLE_NAMES
from stage2_outcome_prediction.classifier import get_decision_tree_for_label, predict
import config


def _cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def get_decision_path_features(tree, x):
    indicator    = tree.decision_path(x.reshape(1, -1))
    tree_nodes   = tree.tree_
    return [int(tree_nodes.feature[n]) for n in indicator.indices if tree_nodes.feature[n] >= 0]


def attribute_premises_dt(x, tree, premise_embeddings, top_k=3):
    """Attribution via decision tree path features."""
    feature_indices = get_decision_path_features(tree, x)
    if len(premise_embeddings) == 0 or len(feature_indices) == 0:
        return []
    scores = np.zeros(len(premise_embeddings))
    for feat_idx in set(feature_indices):
        if feat_idx >= x.shape[0]:
            continue
        direction = np.zeros(x.shape[0])
        direction[feat_idx] = 1.0
        for p_idx, p_emb in enumerate(premise_embeddings):
            scores[p_idx] += abs(x[feat_idx]) * _cosine_similarity(p_emb, direction)
    top = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


def attribute_premises_svm(x, svm, premise_embeddings, top_k=3):
    """Attribution via SVM weight vector: project each premise onto the
    classifier's decision hyperplane to measure how much it 'pushes'
    the prediction toward violation."""
    if len(premise_embeddings) == 0:
        return []
    w = svm.coef_[0]
    # trim to embedding dim — coef_ includes the fallback flag appended by prepare_split
    dim = premise_embeddings.shape[1]
    w = w[:dim]
    scores = np.array([_cosine_similarity(p_emb, w) for p_emb in premise_embeddings])
    top = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


def attribute_premises_generic(case_embedding, premise_embeddings, top_k=3):
    """Fallback: cosine similarity between case embedding and each premise."""
    if len(premise_embeddings) == 0:
        return []
    scores = np.array([_cosine_similarity(case_embedding, p) for p in premise_embeddings])
    top = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


def embed_individual_premises(premises, embedder):
    if not premises:
        return np.empty((0, embedder.dim))
    return embedder.model.encode([p["sentence"] for p in premises], convert_to_numpy=True)


def trace_verdict(case, x, clf, embedder, top_k=3):
    premises   = case.get("premises", [])
    y_pred     = clf.predict(x.reshape(1, -1))[0]
    y_true     = np.array(case.get("labels_binary", [0] * config.NUM_LABELS))
    prem_embs  = embed_individual_premises(premises, embedder)

    justifications = []
    for label_idx, violated in enumerate(y_pred):
        if not violated:
            continue
        supporting = []
        if len(premises) > 0:
            estimator = clf.estimators_[label_idx]
            if isinstance(estimator, LinearSVC):
                ranked = attribute_premises_svm(x, estimator, prem_embs, top_k)
            elif hasattr(estimator, 'tree_'):
                ranked = attribute_premises_dt(x, estimator, prem_embs, top_k)
            else:
                ranked = attribute_premises_generic(x, prem_embs, top_k)

            for p_idx, score in ranked:
                p = premises[p_idx]
                supporting.append({
                    "sentence":     p["sentence"],
                    "paragraph_id": p["paragraph_id"],
                    "sentence_id":  p["sentence_id"],
                    "confidence":   p["confidence"],
                    "attribution":  round(score, 5),
                })
        justifications.append({"article": ARTICLE_NAMES[label_idx], "supporting_premises": supporting})

    return {
        "case_id":            case.get("case_id", -1),
        "predicted_articles": [ARTICLE_NAMES[i] for i, v in enumerate(y_pred) if v == 1],
        "true_articles":      [ARTICLE_NAMES[i] for i, v in enumerate(y_true) if v == 1],
        "justifications":     justifications,
    }


def trace_batch(cases, X, clf, embedder, top_k=3):
    return [trace_verdict(case, x, clf, embedder, top_k) for case, x in zip(cases, X)]


def print_justified_verdict(verdict, max_premises=3):
    print("\n" + "═" * 70)
    print(f"  Case ID  : {verdict['case_id']}")
    print(f"  TRUE     : {verdict['true_articles'] or ['No violation']}")
    print(f"  PREDICTED: {verdict['predicted_articles'] or ['No violation']}")
    print("─" * 70)
    if not verdict["justifications"]:
        print("  No violations predicted.")
        return
    for j in verdict["justifications"]:
        print(f"\n  ▶ {j['article']}")
        if not j["supporting_premises"]:
            print("    (No supporting premises identified)")
            continue
        for rank, p in enumerate(j["supporting_premises"][:max_premises], 1):
            print(f"    [{rank}] (attribution={p['attribution']:.4f}) {p['sentence'][:120]}")
    print("═" * 70)
