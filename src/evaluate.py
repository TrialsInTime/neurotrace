import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold


def subject_cv(features, make_model, classes, seed=42):
    # keep only the chosen classes, e.g. ["AD", "CN"]
    data = features[features.label.isin(classes)].reset_index(drop=True)
    X = data.drop(columns=["subject", "label"]).values
    y = data.label.map({c: i for i, c in enumerate(classes)}).values
    groups = data.subject.values

    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    results = []
    for train, test in folds.split(X, y, groups):
        model = make_model()
        model.fit(X[train], y[train])

        probs = pd.DataFrame(model.predict_proba(X[test]))      # one column per class
        probs["subject"] = groups[test]
        probs["true"] = y[test]
        per_subject = probs.groupby("subject").mean()           # average over each person's epochs

        class_cols = list(range(len(classes)))
        per_subject["guess"] = per_subject[class_cols].values.argmax(axis=1)   # most likely class
        per_subject["true"] = per_subject["true"].round().astype(int)
        results.append(per_subject[["true", "guess"]])

    return pd.concat(results)       # one row per subject: true class and guessed class


def repeat_cv(features, make_model, classes, n_seeds=10):
    accs, f1s, recalls = [], [], []
    for seed in range(n_seeds):
        res = subject_cv(features, make_model, classes, seed)
        accs.append((res.true == res.guess).mean())                        # fraction of people correct
        f1s.append(f1_score(res.true, res.guess, average="macro"))         # F1 averaged over classes
        recalls.append(recall_score(res.true, res.guess, average=None))    # fraction correct, per class

    out = {"accuracy": np.mean(accs), "macro_f1": np.mean(f1s)}
    for i, c in enumerate(classes):
        out[f"recall_{c}"] = np.mean([r[i] for r in recalls])
    return out