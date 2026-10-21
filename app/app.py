# Demo: upload a resting-state EEG recording (EEGLAB .set, 19 channels, 10-20 names)
# or pick one from the dataset, optionally mark channels as missing, get a prediction.
# Run from the project folder:  python app/app.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import matplotlib.pyplot as plt
import mne
import gradio as gr

from src.config import CACHE, DATA_ROOT
from src.data import load_meta, set_path
from src.features import BANDS, load_spectra, recording_spectra, bands_from_spectra
from src.models import make_rf
from src.robustness import fill_channels

CLASSES = ["AD", "FTD", "CN"]
NAMES = {"AD": "Alzheimer's disease", "FTD": "Frontotemporal dementia", "CN": "Cognitively normal"}
MODEL_PATH = CACHE / "demo_rf.joblib"

# training data: cleaned recordings, 4 s epochs, band-power features
spec = load_spectra()
CHANNELS = spec["ch_names"]
train_bands = bands_from_spectra(spec["freqs"], spec["power"])

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = make_rf().fit(train_bands, [CLASSES.index(label) for label in spec["label"]])
    joblib.dump(model, MODEL_PATH)

# average band shares of each group, to compare a recording against
n_bands = len(BANDS)
shares = train_bands[:, : len(CHANNELS) * n_bands].reshape(len(train_bands), len(CHANNELS), n_bands).mean(axis=1)
GROUP_SHARES = {c: shares[spec["label"] == c].mean(axis=0) for c in CLASSES}


def predict(upload, subject, missing):
    if upload is not None:
        path = upload
    elif subject:
        path = set_path(subject)
    else:
        raise gr.Error("Upload a .set file or pick a subject.")

    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    absent = [ch for ch in CHANNELS if ch not in raw.ch_names]
    if absent:
        raise gr.Error(f"Recording is missing channels: {', '.join(absent)}")
    raw.pick(CHANNELS)                                                   # same channel order as training

    freqs, power = recording_spectra(raw)
    power = fill_channels(power, [CHANNELS.index(ch) for ch in missing])   # simulate missing channels
    X = bands_from_spectra(freqs, power)
    probs = model.predict_proba(X).mean(axis=0)                          # average over all epochs

    # band shares of this recording vs the group averages
    mine = X[:, : len(CHANNELS) * n_bands].reshape(len(X), len(CHANNELS), n_bands).mean(axis=(0, 1))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    width = 0.2
    pos = np.arange(n_bands)
    ax.bar(pos - 1.5 * width, mine, width, label="this recording", color="black")
    for i, c in enumerate(CLASSES):
        ax.bar(pos + (i - 0.5) * width, GROUP_SHARES[c], width, label=f"{c} average")
    ax.set_xticks(pos)
    ax.set_xticklabels(list(BANDS))
    ax.set_ylabel("share of total power")
    ax.legend()
    plt.tight_layout()

    summary = f"{len(X)} epochs of 4 s analysed"
    if missing:
        summary += f"; {len(missing)} channel(s) treated as missing"
    return {NAMES[c]: float(p) for c, p in zip(CLASSES, probs)}, fig, summary


subjects = list(load_meta().participant_id) if (DATA_ROOT / "participants.tsv").exists() else []

demo = gr.Interface(
    fn=predict,
    inputs=[gr.File(label="EEG recording (.set)", file_types=[".set"], type="filepath"),
            gr.Dropdown(subjects, label="...or a subject from ds004504", value=None),
            gr.CheckboxGroup(CHANNELS, label="Treat these channels as missing")],
    outputs=[gr.Label(label="Prediction"), gr.Plot(label="Band power"), gr.Textbox(label="Details")],
    title="neurotrace: EEG-based dementia screening (research demo)",
    description="Random forest on relative band power (cleaned, 4 s epochs). "
                "Not a clinical tool. Note: the dataset subjects were used for training, "
                "so predictions on them are optimistic.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
