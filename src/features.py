import numpy as np
import pandas as pd
import mne
from scipy.signal import welch

from .data import load_meta, load_raw

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}

def spectra(data, fs):
    # data has shape (n_epochs, n_channels, n_samples)
    # power spectrum of every channel in every epoch, kept from 1 to 45 Hz (0.5 Hz steps)
    freqs, power = welch(data, fs=fs, nperseg=int(2 * fs), axis=-1)
    use = (freqs >= 1) & (freqs <= 45)
    return freqs[use], power[..., use]


def bands_from_spectra(freqs, power):
    # power has shape (n_epochs, n_channels, n_freqs)
    total = power.sum(axis=-1)                       # total power from 1 to 45 Hz

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


def band_features(data, fs):
    return bands_from_spectra(*spectra(data, fs))

def feature_names(ch_names):
    # same order as band_features: channel by channel, five bands each, then the ratios
    names = [f"{ch}_{band}" for ch in ch_names for band in BANDS]
    names += [f"{ch}_theta/alpha" for ch in ch_names]
    return names


def recording_spectra(raw, duration=4.0):
    # cut one recording into fixed-length epochs and return the spectrum of each
    epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True, verbose="ERROR")
    return spectra(epochs.get_data(), raw.info["sfreq"])


def build_spectra(derivative=True, duration=4.0):
    # spectra of every epoch of every subject, plus who each epoch belongs to
    meta = load_meta()
    powers, subjects, labels = [], [], []
    for sid, label in zip(meta.participant_id, meta.label):
        raw = load_raw(sid, derivative)                                   # one recording
        freqs, power = recording_spectra(raw, duration)                   # (n_epochs, 19, n_freqs)
        powers.append(power.astype(np.float32))
        subjects += [sid] * len(power)
        labels += [label] * len(power)
    return {"power": np.concatenate(powers), "freqs": freqs, "ch_names": raw.ch_names,
            "subject": np.array(subjects), "label": np.array(labels)}


def feature_table(spec):
    # band-power features (one row per epoch) from the output of build_spectra
    X = bands_from_spectra(spec["freqs"], spec["power"])
    table = pd.DataFrame(X, columns=feature_names(spec["ch_names"]))
    table.insert(0, "subject", spec["subject"])                          # who it came from
    table.insert(1, "label", spec["label"])                              # AD, CN or FTD
    return table


def build_feature_table(derivative=True, duration=4.0):
    return feature_table(build_spectra(derivative, duration))


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

def load_spectra(derivative=True, duration=4.0):
    # build_spectra, but saved in cache/ the first time and reloaded after that
    from .config import CACHE
    path = CACHE / f"spectra_{'deriv' if derivative else 'raw'}_{duration:g}s.npz"
    if not path.exists():
        np.savez_compressed(path, **build_spectra(derivative, duration))
    saved = np.load(path)
    return {key: saved[key] for key in saved.files} | {"ch_names": list(saved["ch_names"])}


# ---------- extra features for separating FTD from AD ----------

def region_index(ch_names):
    # positions of each region's channels in the channel list
    return {region: [list(ch_names).index(ch) for ch in channels] for region, channels in REGIONS.items()}


def extra_spectral_features(freqs, power, ch_names):
    # power has shape (n_epochs, n_channels, n_freqs); returns a table of new features
    regions = region_index(ch_names)
    reg = np.stack([power[:, idx].mean(axis=1) for idx in regions.values()], axis=1)   # (n_epochs, 5, n_freqs)
    out = {}

    # peak alpha frequency: where the 6-13 Hz peak sits (it slows down with disease)
    alpha = (freqs >= 6) & (freqs <= 13)
    paf = freqs[alpha][reg[..., alpha].argmax(axis=-1)]
    # spectral entropy: how spread out the spectrum is (0 = one peak, 1 = flat)
    p = reg / reg.sum(axis=-1, keepdims=True)
    entropy = -(p * np.log(p)).sum(axis=-1) / np.log(p.shape[-1])
    for i, region in enumerate(regions):
        out[f"{region}_peak_alpha"] = paf[:, i]
        out[f"{region}_entropy"] = entropy[:, i]

    # front vs back: FTD mainly affects the front of the brain, AD the back
    names = list(regions)
    front = reg[:, names.index("frontal")]
    back = (reg[:, names.index("parietal")] + reg[:, names.index("occipital")]) / 2
    for band, (low, high) in BANDS.items():
        in_band = (freqs >= low) & (freqs < high)
        out[f"front/back_{band}"] = np.log(front[:, in_band].sum(axis=-1) / back[:, in_band].sum(axis=-1))
    return pd.DataFrame(out)


COH_BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}

def coherence_features(data, fs, ch_names):
    # connectivity: how strongly each pair of regions moves together, per band
    from itertools import combinations
    from scipy.signal import coherence
    regions = region_index(ch_names)
    reg = np.stack([data[:, idx].mean(axis=1) for idx in regions.values()], axis=1)     # (n_epochs, 5, n_samples)
    pairs = list(combinations(range(len(regions)), 2))
    freqs, coh = coherence(reg[:, [i for i, _ in pairs]], reg[:, [j for _, j in pairs]],
                           fs=fs, nperseg=int(fs), axis=-1)                              # (n_epochs, 10, n_freqs)
    names = list(regions)
    out = {}
    for band, (low, high) in COH_BANDS.items():
        in_band = (freqs >= low) & (freqs < high)
        for k, (i, j) in enumerate(pairs):
            out[f"coh_{band}_{names[i]}-{names[j]}"] = coh[:, k, in_band].mean(axis=-1)
    return pd.DataFrame(out)


def load_coherence(derivative=True, duration=4.0):
    # coherence features for every epoch (same order as load_spectra), cached in cache/
    from .config import CACHE
    path = CACHE / f"coherence_{'deriv' if derivative else 'raw'}_{duration:g}s.pkl"
    if not path.exists():
        tables = []
        for sid in load_meta().participant_id:
            raw = load_raw(sid, derivative)
            epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True, verbose="ERROR")
            tables.append(coherence_features(epochs.get_data(), raw.info["sfreq"], raw.ch_names))
        pd.concat(tables, ignore_index=True).to_pickle(path)
    return pd.read_pickle(path)


def extended_table(spec, coh):
    # band-power features + the extra spectral features + connectivity
    extra = extra_spectral_features(spec["freqs"], spec["power"], spec["ch_names"])
    return pd.concat([feature_table(spec), extra, coh], axis=1)


def per_subject(table):
    # one row per person: the median of each feature over all their epochs
    return table.groupby(["subject", "label"], sort=False).median().reset_index()


def method_table(spec, coh):
    # our method's features, one row per person:
    # regional band power + peak alpha, entropy, front/back ratios + regional coherence
    extra = extra_spectral_features(spec["freqs"], spec["power"], spec["ch_names"])
    return per_subject(pd.concat([regional_table(feature_table(spec)), extra, coh], axis=1))
