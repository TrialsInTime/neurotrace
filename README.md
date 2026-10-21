
# neurotrace

EEG-based Alzheimer's detection with preprocessing comparison, missing-channel robustness testing, explainability and edge-deployment benchmarking.

> Work in progress: course project (VIT).

## Overview

The system takes a resting-state EEG recording and produces:

- a predicted diagnostic category (cognitively normal / AD / FTD)
- a prediction confidence score
- a robustness profile showing how accuracy degrades as EEG channels are dropped or corrupted
- a visual explanation of which channels, frequency bands or time windows drove the prediction
- a latency and memory benchmark on constrained hardware

## Dataset

[OpenNeuro ds004504](https://openneuro.org/datasets/ds004504): resting-state, eyes-closed EEG from 88 participants (AD, frontotemporal dementia, healthy controls), recorded with 19 scalp electrodes and provided in BIDS format with raw and preprocessed versions.

The data is **not** stored in this repository. Download it from OpenNeuro and place it in a folder called `dataset/` next to the `neurotrace/` folder, or set the `NEUROTRACE_DATA` environment variable to its location.

## Pipeline

1. Preprocessing comparison: raw vs cleaned recordings (ICA + artifact removal), epoch length 2 / 4 / 8 s
2. Feature extraction: relative band power (delta, theta, alpha, beta, gamma) and theta/alpha ratio per channel
3. Classical baseline (logistic regression / random forest)
4. Deep classifier (small 1D CNN on each epoch's 19-channel power spectrum)
5. Subject-level stratified cross-validation (no epoch leakage between train and test)
6. Missing-channel robustness testing
7. Explainability (SHAP, saliency / Grad-CAM)
8. Edge-deployment benchmarking
9. Demo interface

## Repository structure

```
configs/     preprocessing configurations
src/         data loading, preprocessing, features, models, robustness, explainability, benchmarking
notebooks/   Colab notebooks that call code in src/
results/     small result tables and figures
app/         demo interface
```

## Notebooks

| Notebook | What it does |
|---|---|
| `00_explore` | Look at raw signals and spectra |
| `01_features` | Build the band-power feature table |
| `02_baseline` | Logistic regression and random forest |
| `03_preprocessing` | Raw vs cleaned, 2 / 4 / 8 s epochs |
| `04_cnn` | 1D CNN on spectra |
| `05_robustness` | Accuracy with missing channels |
| `06_explainability` | SHAP (random forest) and saliency (CNN) |
| `07_benchmark` | Model size and single-core latency |
| `08_our_method` | Our method vs the baseline, step-by-step ablation |

Spectra are cached in `cache/` the first time they are computed (a few minutes per setting).

## Demo

```
python app/app.py
```

Upload an EEGLAB `.set` recording (19 channels, 10-20 names) or pick a dataset subject, optionally mark channels as missing, and get class probabilities plus a band-power chart. The demo model is trained on all dataset subjects, so predictions on those subjects are optimistic.

## Setup (Colab)

```python
!git clone https://github.com/TrialsInTime/neurotrace.git
%cd neurotrace
!pip install -q -r requirements.txt
```

## Results

### Our method: better FTD detection

The baselines call most FTD subjects AD. Our method keeps the same random forest but changes the input:

1. **One row per person** (median of each feature over their epochs) instead of thousands of epochs
2. **Regional features** (5 brain regions instead of 19 channels)
3. **FTD-targeted features**: peak alpha frequency, spectral entropy, front-vs-back power ratios, and coherence between regions

AD vs FTD vs CN, 10 shuffles (`results/our_method_ablation.csv`):

| Method | Accuracy | Macro F1 | FTD recall |
|---|---|---|---|
| Baseline: RF on epochs, channel band power | **0.644** | 0.588 | 0.217 |
| + one row per person | 0.551 | 0.530 | 0.335 |
| + regional features | 0.550 | 0.535 | 0.378 |
| + peak alpha, entropy, front/back ratios | 0.597 | 0.581 | 0.422 |
| **+ coherence (our method)** | 0.624 | **0.604** | **0.409** |
| Our method with a two-step classifier (healthy vs dementia, then AD vs FTD) | 0.599 | 0.566 | 0.322 |

Per task (`results/our_method_vs_baseline.csv`):

| Task | Baseline F1 | Our F1 | Baseline FTD recall | Our FTD recall |
|---|---|---|---|---|
| AD vs FTD vs CN | 0.588 | **0.604** | 0.217 | **0.409** |
| AD vs FTD | 0.519 | **0.566** | 0.187 | **0.487** |
| FTD vs CN | 0.764 | **0.785** | 0.652 | **0.726** |
| AD vs CN | **0.820** | 0.783 | - | - |

Our method roughly **doubles FTD recall** (0.22 to 0.41 in the 3-class task, 0.19 to 0.49 for AD vs FTD) and has the best macro F1 on every task involving FTD. The cost is AD recall (0.78 to 0.62), because AD is no longer the default answer, plus about 2 points of overall accuracy. On AD vs CN alone the baseline stays better. The most important features are the new ones: parietal-occipital coherence and front-vs-back beta and alpha ratios (`results/our_method_features.png`). The two-step classifier did not help.

### Other experiments

All accuracies are **per subject** (a subject's epoch predictions are averaged), with 5-fold cross-validation that never splits a subject between train and test, averaged over several shuffles.

**Models** (cleaned data, 4 s epochs; `results/model_comparison.csv`)

| Task | LogReg | Random forest | 1D CNN |
|---|---|---|---|
| AD vs CN | 0.79 | **0.82** | 0.68 |
| AD vs FTD vs CN | 0.57 | **0.64** | 0.51 |

FTD is the hard class: only 22-29 % of FTD subjects are recognised, most are called AD. With only 88 subjects the CNN overfits and does worse than the classical models.

**Preprocessing** (`results/preprocessing_comparison.csv`): the differences are small (within about +/- 4 points) and inside the shuffle-to-shuffle noise. Cleaning gives no clear gain for band-power features; 8 s epochs help logistic regression slightly on AD vs CN (0.84).

**Missing channels** (AD vs CN; `results/robustness_AD-CN.png`): missing channels are filled with the average of the remaining ones. The random forest is the most robust: 0.82 with all channels, 0.76 with 2 to 6 missing, 0.71 with 8 of 19 missing. Logistic regression drops to 0.64 and the CNN to 0.58.

**Explainability** (`results/shap_*.png`, `results/cnn_saliency.png`): the theta/alpha ratio is the most important feature group for the random forest, and the occipital and left temporal-parietal channels matter most. This agrees with the known slowing of posterior alpha rhythm in Alzheimer's disease.

**Edge benchmark** (AD vs CN, one CPU core; `results/edge_benchmark.csv`)

| Model | Accuracy | Size | Time per 4 s epoch (features + prediction) |
|---|---|---|---|
| LogReg | 0.79 | 4 KB | 1.1 ms |
| Random forest (300 trees) | 0.82 | 57 MB | 14.8 ms |
| Random forest (50 trees, depth 8) | 0.80 | 1.5 MB | 3.3 ms |
| 1D CNN | 0.68 | 54 KB | 1.0 ms |

All models are far faster than real time. The small random forest keeps almost all of the accuracy at 1/38 of the size, making it the best choice for constrained hardware.

## Limitations

- Small, single-site cohort (88 subjects) and only 19 channels
- Connectivity and microstate features were not explored
- FTD remains the hardest class: even with our method, about 4 in 10 FTD subjects are called AD
- Cross-sectional data, so no progression modeling
- Not a clinical tool

## License

MIT




