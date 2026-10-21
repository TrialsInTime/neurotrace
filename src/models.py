import numpy as np
import torch
from torch import nn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def make_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def make_rf():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)


class TwoStep:
    # step 1: healthy vs dementia (any kind); step 2: AD vs FTD, trained on dementia subjects only
    # y uses the class numbers from subject_cv; `healthy` is the number of the CN class
    def __init__(self, make_model, healthy):
        self.make_model, self.healthy = make_model, healthy

    def fit(self, X, y):
        sick = y != self.healthy
        self.step1 = self.make_model().fit(X, sick.astype(int))
        self.step2 = self.make_model().fit(X[sick], y[sick])
        self.n_classes = int(y.max()) + 1
        return self

    def predict_proba(self, X):
        p_sick = self.step1.predict_proba(X)[:, 1]
        out = np.zeros((len(X), self.n_classes))
        out[:, self.healthy] = 1 - p_sick
        for k, c in enumerate(self.step2.classes_):          # split the dementia probability between AD and FTD
            out[:, c] = p_sick * self.step2.predict_proba(X)[:, k]
        return out


class SpectraCNN(nn.Module):
    # 1D CNN that slides along frequency; the 19 EEG channels are the input channels
    def __init__(self, n_channels=19, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(32, n_classes),
        )

    def forward(self, x):            # x: (batch, channels, freqs)
        return self.net(x)


class CNNClassifier:
    # wraps SpectraCNN so it has the same fit / predict_proba as the sklearn models
    # X rows are flattened spectra: n_channels * n_freqs values per epoch
    def __init__(self, n_channels=19, n_epochs=8, lr=1e-3, seed=0):
        self.n_channels, self.n_epochs, self.lr, self.seed = n_channels, n_epochs, lr, seed

    def prepare(self, X):
        x = X.reshape(len(X), self.n_channels, -1)
        x = np.log10(x / x.sum(axis=-1, keepdims=True))             # relative power, log scale
        return torch.tensor((x - self.mean) / self.std, dtype=torch.float32)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        x = X.reshape(len(X), self.n_channels, -1)
        x = np.log10(x / x.sum(axis=-1, keepdims=True))
        self.mean, self.std = x.mean(axis=0), x.std(axis=0) + 1e-6  # from training data only
        x, y = self.prepare(X), torch.tensor(y)

        n_classes = int(y.max()) + 1
        self.model = SpectraCNN(self.n_channels, n_classes)
        weights = len(y) / (n_classes * torch.bincount(y).float())   # like class_weight="balanced"
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        for _ in range(self.n_epochs):
            for batch in torch.randperm(len(y)).split(256):
                opt.zero_grad()
                loss_fn(self.model(x[batch]), y[batch]).backward()
                opt.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(self.prepare(X)), dim=1).numpy()


def make_cnn():
    return CNNClassifier()


def spectra_table(spec):
    # flattened spectra as a table with subject / label columns, ready for subject_cv
    import pandas as pd
    table = pd.DataFrame(spec["power"].reshape(len(spec["power"]), -1))
    table.insert(0, "subject", spec["subject"])
    table.insert(1, "label", spec["label"])
    return table
