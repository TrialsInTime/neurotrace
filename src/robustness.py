import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .features import bands_from_spectra


def fill_channels(power, lost):
    # replace the spectra of the lost channels with the average spectrum
    # of the channels that are left (a crude interpolation)
    power = power.copy()
    if len(lost) == 0:
        return power
    kept = np.setdiff1d(np.arange(power.shape[1]), lost)
    power[:, lost] = power[:, kept].mean(axis=1, keepdims=True)
    return power


def drop_channels(power, k, rng):
    # pretend k random channels are missing
    return fill_channels(power, rng.choice(power.shape[1], size=k, replace=False))


def as_bands(spec):
    # model input = band-power features (for LogReg / RandomForest)
    return lambda power: bands_from_spectra(spec["freqs"], power)


def as_spectra(power):
    # model input = flattened spectra (for the CNN)
    return power.reshape(len(power), -1)


def robustness_cv(spec, make_model, to_input, classes, ks=(0, 1, 2, 4, 6, 8), repeats=5, seed=0):
    # train on complete recordings, then test with k channels missing
    keep = np.isin(spec["label"], classes)
    power, groups = spec["power"][keep], spec["subject"][keep]
    y = pd.Series(spec["label"][keep]).map({c: i for i, c in enumerate(classes)}).values

    rng = np.random.default_rng(seed)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    correct = {k: [] for k in ks}
    for train, test in folds.split(power, y, groups):
        model = make_model()
        model.fit(to_input(power[train]), y[train])

        for k in ks:
            for _ in range(repeats if k else 1):                   # different lost channels each time
                probs = pd.DataFrame(model.predict_proba(to_input(drop_channels(power[test], k, rng))))
                probs["subject"], probs["true"] = groups[test], y[test]
                per_subject = probs.groupby("subject").mean()     # average over each person's epochs
                guess = per_subject[list(range(len(classes)))].values.argmax(axis=1)
                correct[k].append(guess == per_subject["true"].values)

    # fraction of subjects correct, for each number of missing channels
    return pd.Series({k: np.concatenate(v).mean() for k, v in correct.items()}, name="accuracy")
