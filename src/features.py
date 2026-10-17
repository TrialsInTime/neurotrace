import numpy as np
import pandas as pd
import mne
from scipy.signal import welch

from .data import load_meta, load_raw

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}

def band_features(data, fs):
    # data has shape (n_epochs, n_channels, n_samples)

    # power spectrum of every channel in every epoch
    freqs, power = welch(data, fs=fs, nperseg=int(2 * fs), axis=-1)

    # total power from 1 to 45 Hz
    use = (freqs >= 1) & (freqs <= 45)
    total = power[..., use].sum(axis=-1)

    # share of the total in each band
    shares = []
    for low, high in BANDS.values():
        in_band = (freqs >= low) & (freqs < high)
        shares.append(power[..., in_band].sum(axis=-1) / total)
    shares = np.stack(shares, axis=-1)               # (n_epochs, n_channels, 5)

    # theta/alpha ratio for every channel
    ratio = shares[..., 1] / shares[..., 2]          # (n_epochs, n_channels)

    # flatten into one row of features per epoch
    return np.concatenate([shares.reshape(len(shares), -1), ratio], axis=1)

def feature_names(ch_names):
    # same order as band_features: channel by channel, five bands each, then the ratios
    names = [f"{ch}_{band}" for ch in ch_names for band in BANDS]
    names += [f"{ch}_theta/alpha" for ch in ch_names]
    return names


def build_feature_table(derivative=True):
    meta = load_meta()
    tables = []
    for sid, label in zip(meta.participant_id, meta.label):
        raw = load_raw(sid, derivative)                                   # one recording
        epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True, verbose="ERROR")
        X = band_features(epochs.get_data(), raw.info["sfreq"])           # one row per epoch
        table = pd.DataFrame(X, columns=feature_names(raw.ch_names))
        table.insert(0, "subject", sid)                                   # who it came from
        table.insert(1, "label", label)                                   # AD, CN or FTD
        tables.append(table)
    return pd.concat(tables, ignore_index=True)      


REGIONS = {
    "frontal":   ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "central":   ["C3", "C4", "Cz"],
    "temporal":  ["T3", "T4", "T5", "T6"],
    "parietal":  ["P3", "P4", "Pz"],
    "occipital": ["O1", "O2"],
}

def regional_table(features):
    out = features[["subject", "label"]].copy()
    for region, channels in REGIONS.items():
        for band in BANDS:
            columns = [f"{ch}_{band}" for ch in channels]
            out[f"{region}_{band}"] = features[columns].mean(axis=1)      # average over the region's channels
        out[f"{region}_theta/alpha"] = out[f"{region}_theta"] / out[f"{region}_alpha"]
    return out                     # stack all subjects