import pandas as pd
import mne
from .config import DATA_ROOT

LABELS = {"A": "AD", "C": "CN", "F": "FTD"}

def load_meta():
    meta = pd.read_csv(DATA_ROOT / "participants.tsv", sep="\t")
    meta["label"] = meta["Group"].map(LABELS)
    return meta

def set_path(sid, derivative=True):
    base = DATA_ROOT / "derivatives" / sid / "eeg" if derivative else DATA_ROOT / sid / "eeg"
    return base / f"{sid}_task-eyesclosed_eeg.set"

def load_raw(sid, derivative=True):
    return mne.io.read_raw_eeglab(set_path(sid, derivative), preload=True, verbose="ERROR")