"""
Feature-selection stability analysis: for each outer CV fold, fit LASSO on
the training split and record which features get non-zero coefficients.
A feature selected in most/all folds is a stable signal; a feature selected
in only one or two folds is likely fold-specific noise. This is the
diagnostic that explains *why* radiomics under/over-performs, not just
*that* it does.

Usage:
    python scripts/05_feature_stability.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


def load_merged_with_names():
    clinical = pd.read_csv(config.CLINICAL_CLEAN_CSV)
    features = pd.read_csv(config.FEATURES_CSV)
    if "roi_used" in features.columns:
        features = features.drop(columns=["roi_used"])

    merged = clinical.merge(features, on="PatientID", how="inner")
    y = merged["Outcome"].astype(int).values

    clinical_cols = [c for c in clinical.columns if c not in ("PatientID", "Outcome")]
    radiomics_cols = [c for c in features.columns if c != "PatientID"]

    blocks = {
        "clinical": (merged[clinical_cols].values, clinical_cols),
        "radiomics": (merged[radiomics_cols].values, radiomics_cols),
        "combined": (merged[clinical_cols + radiomics_cols].values, clinical_cols + radiomics_cols),
    }
    return blocks, y


def lasso_pipeline():
    return Pipeline([
        ("var", VarianceThreshold()),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear",
                                    max_iter=5000, random_state=config.RANDOM_STATE)),
    ])


def selection_frequency(X, feature_names, y, n_folds=None):
    """
    Run LASSO with the same GridSearchCV setup as 03_ml_pipeline.py across
    outer folds, and for each fold record which original feature names
    survive both the VarianceThreshold step and end up with a non-zero
    LASSO coefficient. Returns a DataFrame: feature, times_selected, frac.
    """
    n_folds = n_folds or config.OUTER_FOLDS
    outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)
    grid = {"clf__C": [0.01, 0.05, 0.1, 0.5, 1.0]}

    selected_counts = {name: 0 for name in feature_names}
    n_folds_run = 0

    for fold_i, (train_idx, test_idx) in enumerate(outer.split(X, y), 1):
        pipe = lasso_pipeline()
        inner = StratifiedKFold(n_splits=config.INNER_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1)
        search.fit(X[train_idx], y[train_idx])
        best_pipe = search.best_estimator_

        # Which original columns survived VarianceThreshold, in order:
        var_mask = best_pipe.named_steps["var"].get_support()
        surviving_names = [n for n, keep in zip(feature_names, var_mask) if keep]

        coefs = best_pipe.named_steps["clf"].coef_.ravel()
        nonzero_mask = np.abs(coefs) > 1e-8
        selected_names = [n for n, nz in zip(surviving_names, nonzero_mask) if nz]

        for name in selected_names:
            selected_counts[name] += 1
        n_folds_run += 1
        print(f"  fold {fold_i}: {len(selected_names)} features selected "
              f"(best C = {search.best_params_['clf__C']})")

    freq_df = pd.DataFrame([
        {"feature": name, "times_selected": count, "fraction_of_folds": count / n_folds_run}
        for name, count in selected_counts.items()
    ]).sort_values("times_selected", ascending=False)
    return freq_df


def main():
    if not config.CLINICAL_CLEAN_CSV.exists() or not config.FEATURES_CSV.exists():
        print("ERROR: run 01_extract_radiomics.py and 02_prepare_clinical.py first.")
        sys.exit(1)

    blocks, y = load_merged_with_names()
    out_dir = config.OUTPUT_DIR / "feature_stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    for block_name in ("radiomics", "combined"):
        X, feature_names = blocks[block_name]
        print(f"\n=== Feature stability: {block_name} block "
              f"({X.shape[1]} features, {X.shape[0]} patients) ===")
        freq_df = selection_frequency(X, feature_names, y)

        out_path = out_dir / f"{block_name}_feature_stability.csv"
        freq_df.to_csv(out_path, index=False)

        n_never = (freq_df["times_selected"] == 0).sum()
        n_always = (freq_df["fraction_of_folds"] == 1.0).sum()
        print(f"  Never selected in any fold: {n_never} / {len(freq_df)}")
        print(f"  Selected in EVERY fold (stable): {n_always}")
        if n_always > 0:
            print(f"  Stable features:\n"
                  f"{freq_df[freq_df['fraction_of_folds'] == 1.0][['feature']].to_string(index=False)}")
        print(f"  Top 10 most frequently selected:\n"
              f"{freq_df.head(10)[['feature', 'times_selected']].to_string(index=False)}")
        print(f"  Saved full table to {out_path}")


if __name__ == "__main__":
    main()