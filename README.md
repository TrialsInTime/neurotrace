
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

The data is **not** stored in this repository. Download it from OpenNeuro and place it under `data/ds004504/` (or point the config at your own location).

## Pipeline

1. Preprocessing with several configurations (referencing, artifact rejection, epoch length)
2. Feature extraction: band power, connectivity (microstates optional)
3. Classical baseline (logistic regression / random forest)
4. Deep classifier (1D CNN)
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

## Setup (Colab)

```python
!git clone https://github.com/TrialsInTime/neurotrace.git
%cd neurotrace
!pip install -q -r requirements.txt
```

## Results

_To be added._

## Limitations

- Small, single-site cohort (88 subjects) and only 19 channels
- Cross-sectional data, so no progression modeling
- Not a clinical tool

## License

MIT




