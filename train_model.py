"""
train_model.py
==============
Standalone training script — no internal imports.
Used by Render build command: python train_model.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR / "model"
GENRES     = ["blues","classical","country","disco","hiphop",
               "jazz","metal","pop","reggae","rock"]
FEATURE_DIM = 142
N_PER_CLASS = 60
SEED        = 42

# ── Generate synthetic demo data ──────────────────────────────────────────────
print("Generating demo training data...")
rng = np.random.default_rng(SEED)
X_list, y_list = [], []

for i, genre in enumerate(GENRES):
    mean    = rng.uniform(-2, 2, FEATURE_DIM) + (i * 0.5)
    cov     = np.eye(FEATURE_DIM) * rng.uniform(0.5, 1.5)
    samples = rng.multivariate_normal(mean, cov, N_PER_CLASS).astype(np.float32)
    X_list.append(samples)
    y_list.extend([genre] * N_PER_CLASS)

X     = np.vstack(X_list)
y_raw = np.array(y_list)
print(f"Dataset: {X.shape[0]} samples, {len(GENRES)} genres")

# ── Encode + scale ────────────────────────────────────────────────────────────
le     = LabelEncoder()
y      = le.fit_transform(y_raw)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
)
clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
print("Training RandomForestClassifier...")
clf.fit(X_train, y_train)

acc = (clf.predict(X_test) == y_test).mean()
print(f"Test accuracy: {acc*100:.1f}%")

# ── Save artefacts ────────────────────────────────────────────────────────────
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(clf,    MODEL_DIR / "genre_model.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(le,     MODEL_DIR / "label_encoder.pkl")
print(f"Model artefacts saved to {MODEL_DIR}")
print("Build complete!")
