"""
ML Demo: Decision Trees, KNN, Cross-Validation, and AUC
Topics: Nearest Neighbor & Decision Trees, Overfitting & Cross-Validation,
        Performance Evaluation & AUC
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA


# ── 1. Load & inspect data ────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
class_names   = data.target_names        # ['malignant', 'benign']

print(f"Dataset : Breast Cancer Wisconsin")
print(f"Samples : {X.shape[0]}  |  Features : {X.shape[1]}")
print(f"Classes : {class_names}  |  Counts: {np.bincount(y)}\n")


# ── 2. Preprocessing ──────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ── 3. PCA for 2-D visualisation ──────────────────────────────────────────────
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}\n")


# ── 4. Models ─────────────────────────────────────────────────────────────────
models = {
    "KNN (k=5)":       KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":   DecisionTreeClassifier(max_depth=4, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 50)
print(f"{'Model':<20} {'AUC (mean±std)':>20}")
print("=" * 50)

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")
    cv_results[name] = scores
    print(f"{name:<20} {scores.mean():.4f} ± {scores.std():.4f}")

print("=" * 50, "\n")


# ── 5. Fit on full data for plots ─────────────────────────────────────────────
for model in models.values():
    model.fit(X_scaled, y)


# ── 6. Figure layout ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("ML Demo: KNN & Decision Tree — Breast Cancer Dataset", fontsize=14)

# 6a. PCA scatter
ax = axes[0, 0]
for label, color in zip([0, 1], ["#e74c3c", "#2ecc71"]):
    mask = y == label
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, alpha=0.5,
               label=class_names[label], s=20)
ax.set_title("PCA — 2D projection")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.legend()

# 6b. Cross-validation AUC box plots
ax = axes[0, 1]
ax.boxplot(cv_results.values(), labels=cv_results.keys(), patch_artist=True)
ax.set_title("5-Fold Cross-Validation AUC")
ax.set_ylabel("ROC AUC")
ax.set_ylim(0.85, 1.01)
ax.axhline(0.95, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

# 6c. KNN: effect of k on CV AUC
ax = axes[0, 2]
ks  = range(1, 21)
aucs = [cross_val_score(KNeighborsClassifier(n_neighbors=k),
                        X_scaled, y, cv=cv, scoring="roc_auc").mean()
        for k in ks]
ax.plot(ks, aucs, "o-", color="#3498db")
ax.set_title("KNN — AUC vs. k (bias/variance trade-off)")
ax.set_xlabel("k (number of neighbours)")
ax.set_ylabel("Mean CV AUC")