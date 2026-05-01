import numpy as np
import joblib
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import config


def train_classifier(X_train, y_train):
    """Train a multi-output classifier based on config.CLASSIFIER_TYPE."""
    if config.CLASSIFIER_TYPE == "decision_tree":
        base = DecisionTreeClassifier(
            max_depth=config.DT_MAX_DEPTH,
            min_samples_leaf=config.DT_MIN_SAMPLES,
            criterion=config.DT_CRITERION,
            random_state=config.RANDOM_SEED,
        )
    elif config.CLASSIFIER_TYPE == "svm":
        base = LinearSVC(
            C=config.SVM_C,
            max_iter=config.SVM_MAX_ITER,
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
        )
    elif config.CLASSIFIER_TYPE == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        base = ExplainableBoostingClassifier(
            max_bins=config.EBM_MAX_BINS,
            outer_bags=config.EBM_OUTER_BAGS,
            random_state=config.RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unknown CLASSIFIER_TYPE: '{config.CLASSIFIER_TYPE}'")

    clf = MultiOutputClassifier(base, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


def predict(clf, X):
    """Predict binary labels for each article."""
    return clf.predict(X)


def predict_proba(clf, X):
    """Get probability scores for positive class. Not available for SVM."""
    if not hasattr(clf.estimators_[0], 'predict_proba'):
        raise NotImplementedError(f"{type(clf.estimators_[0]).__name__} does not support predict_proba.")
    probas = clf.predict_proba(X)
    return np.column_stack([p[:, 1] for p in probas])


def save_classifier(clf, path=config.MODEL_CACHE):
    """Save trained classifier to disk."""
    joblib.dump(clf, path)


def load_classifier(path=config.MODEL_CACHE):
    """Load trained classifier from disk."""
    return joblib.load(path)


def get_decision_tree_for_label(clf, label_idx):
    """Get the decision tree for a specific label. Only works for decision tree classifiers."""
    if config.CLASSIFIER_TYPE != "decision_tree":
        raise ValueError(f"Classifier type '{config.CLASSIFIER_TYPE}' is not a decision tree.")
    return clf.estimators_[label_idx]


def decision_tree_rules(clf, label_idx, feature_names=None):
    """Export decision tree rules as text."""
    try:
        tree = get_decision_tree_for_label(clf, label_idx)
        return export_text(tree, feature_names=feature_names, show_weights=True, max_depth=5)
    except ValueError as e:
        return f"(Not available: {e})"


def decision_scores(clf, X):
    """Get raw decision scores per article. Works for SVM (decision_function) and tree/EBM (predict_proba)."""
    n_samples = X.shape[0]
    n_labels = len(clf.estimators_)
    scores = np.zeros((n_samples, n_labels))

    for i, estimator in enumerate(clf.estimators_):
        if hasattr(estimator, "decision_function"):
            scores[:, i] = estimator.decision_function(X)
        elif hasattr(estimator, "predict_proba"):
            scores[:, i] = estimator.predict_proba(X)[:, 1]
        else:
            scores[:, i] = estimator.predict(X)

    return scores


def tune_thresholds(clf, X_val, y_val, n_steps=50):
    """Find per-article thresholds that maximize F1 on validation set."""
    scores = decision_scores(clf, X_val)
    n_labels = scores.shape[1]
    thresholds = np.zeros(n_labels)

    for i in range(n_labels):
        best_f1 = 0.0
        best_threshold = 0.0
        score_range = np.linspace(scores[:, i].min(), scores[:, i].max(), n_steps)

        for threshold in score_range:
            predictions = (scores[:, i] >= threshold).astype(int)
            f1 = f1_score(y_val[:, i], predictions, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        thresholds[i] = best_threshold

    return thresholds


def predict_with_thresholds(clf, X, thresholds):
    """Predict using per-article tuned thresholds instead of default 0."""
    scores = decision_scores(clf, X)
    return (scores >= thresholds).astype(int)


def quick_eval(clf, X_val, y_val):
    """Quick validation evaluation showing macro and micro F1 scores."""
    y_pred = clf.predict(X_val)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_val, y_pred, average="micro", zero_division=0)
    print(f"Val  Macro-F1={macro_f1:.4f}  Micro-F1={micro_f1:.4f}")
    return {"macro_f1": macro_f1, "micro_f1": micro_f1}


if __name__ == "__main__":
    rng     = np.random.default_rng(42)
    X_train = rng.random((200, 384))
    y_train = (rng.random((200, 10)) > 0.85).astype(int)
    clf     = train_classifier(X_train, y_train)
    quick_eval(clf, rng.random((50, 384)), (rng.random((50, 10)) > 0.85).astype(int))
