import numpy as np
import joblib
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import config


def train_classifier(X_train, y_train):
    """Train a multi-output SVM classifier."""
    base = LinearSVC(
        C=config.SVM_C,
        max_iter=config.SVM_MAX_ITER,
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
    )
    clf = MultiOutputClassifier(base, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


def predict(clf, X):
    """Predict binary labels for each article."""
    return clf.predict(X)


def predict_proba(clf, X):
    """Get probability scores for positive class. Not supported by SVM."""
    raise NotImplementedError("SVM does not support predict_proba. Use decision_scores() instead.")


def save_classifier(clf, path=config.MODEL_CACHE):
    """Save trained classifier to disk."""
    joblib.dump(clf, path)


def load_classifier(path=config.MODEL_CACHE):
    """Load trained classifier from disk."""
    return joblib.load(path)




def decision_scores(clf, X):
    """Get raw decision scores per article using SVM decision_function."""
    n_samples = X.shape[0]
    n_labels = len(clf.estimators_)
    scores = np.zeros((n_samples, n_labels))

    for i, estimator in enumerate(clf.estimators_):
        scores[:, i] = estimator.decision_function(X)

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
