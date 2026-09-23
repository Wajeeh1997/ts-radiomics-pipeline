"""
Exploratory data analysis: patient demographics, outcome balance, and
radiomics feature distributions, before modeling.

Usage:
    python scripts/eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    clinical = pd.read_csv(config.CLINICAL_CLEAN_CSV)
    features = pd.read_csv(config.FEATURES_CSV)

    eda_dir = config.OUTPUT_DIR / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    print(f"Clinical table: {clinical.shape[0]} patients, {clinical.shape[1]} columns")
    print(f"Radiomics table: {features.shape[0]} patients, {features.shape[1]} columns")

    merged_ids = set(clinical["PatientID"]) & set(features["PatientID"])
    print(f"Patients present in both tables: {len(merged_ids)}")
    only_clinical = set(clinical["PatientID"]) - set(features["PatientID"])
    only_features = set(features["PatientID"]) - set(clinical["PatientID"])
    if only_clinical:
        print(f"  In clinical only (no usable radiomics): {sorted(only_clinical)}")
    if only_features:
        print(f"  In radiomics only (no usable outcome): {sorted(only_features)}")

    # ---- Outcome balance ----
    print(f"\nOutcome distribution: {clinical['Outcome'].value_counts().to_dict()}")
    fig, ax = plt.subplots(figsize=(4, 4))
    clinical["Outcome"].value_counts().sort_index().plot(
        kind="bar", ax=ax, color=["#4C72B0", "#DD8452"]
    )
    ax.set_xticklabels(["No lung mets (0)", "Lung mets (1)"], rotation=0)
    ax.set_ylabel("Patients")
    ax.set_title("Outcome balance")
    fig.tight_layout()
    fig.savefig(eda_dir / "outcome_balance.png", dpi=150)
    plt.close(fig)

    # ---- Age distribution (if present) ----
    if "Age" in clinical.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        clinical.groupby("Outcome")["Age"].plot(kind="hist", alpha=0.6, ax=ax, legend=True)
        ax.set_xlabel("Age")
        ax.set_title("Age distribution by outcome")
        fig.tight_layout()
        fig.savefig(eda_dir / "age_by_outcome.png", dpi=150)
        plt.close(fig)

    # ---- Missingness in clinical covariates ----
    missing = clinical.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"\nColumns with missing values (top 10):\n{missing.head(10)}")

    # ---- Radiomics feature sanity ----
    feat_cols = [c for c in features.columns if c.startswith("CT_") or c.startswith("PET_")]
    print(f"\nRadiomics feature columns: {len(feat_cols)} "
          f"({sum(c.startswith('CT_') for c in feat_cols)} CT, "
          f"{sum(c.startswith('PET_') for c in feat_cols)} PET)")

    nan_frac = features[feat_cols].isna().mean().sort_values(ascending=False)
    n_with_nans = (nan_frac > 0).sum()
    print(f"Feature columns with any missing values: {n_with_nans} / {len(feat_cols)}")
    if n_with_nans > 0:
        print(f"Worst offenders:\n{nan_frac.head(10)}")

    # Feature variance sanity — near-constant features are useless / risky for modeling
    variances = features[feat_cols].var(numeric_only=True).sort_values()
    near_zero = variances[variances < 1e-8]
    if len(near_zero) > 0:
        print(f"\n{len(near_zero)} features have near-zero variance (uninformative):")
        print(list(near_zero.index)[:10], "..." if len(near_zero) > 10 else "")

    # ---- Feature correlation heatmap (first 30 features, for a quick look) ----
    sample_feats = feat_cols[:30]
    corr = features[sample_feats].corr()
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(sample_feats)))
    ax.set_yticks(range(len(sample_feats)))
    ax.set_xticklabels(sample_feats, rotation=90, fontsize=6)
    ax.set_yticklabels(sample_feats, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature correlation (first 30 radiomics features)")
    fig.tight_layout()
    fig.savefig(eda_dir / "feature_correlation_sample.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved EDA plots to {eda_dir}")


if __name__ == "__main__":
    main()