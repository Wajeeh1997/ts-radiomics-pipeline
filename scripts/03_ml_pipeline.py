"""
Nested cross-validation comparison of clinical-only, radiomics-only, and
combined feature blocks for predicting lung metastasis — same structure as
the thesis's clinical-vs-radiomics-vs-toxicity comparison, retargeted to the
Soft-Tissue-Sarcoma cohort (51 patients — small, so folds are lighter than
the HECKTOR version).

Usage:
    python scripts/03_ml_pipeline.py [--n-perm 200]
"""
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle


MODEL_GRIDS = {
    "LASSO": (
        Pipeline([
            ("var", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(penalty="l1", solver="liblinear",
                                        max_iter=5000, random_state=config.RANDOM_STATE)),
        ]),
        {"clf__C": [0.01, 0.05, 0.1, 0.5, 1.0]},
    ),
    "RandomForest": (
        Pipeline([
            ("var", VarianceThreshold()),
            ("clf", RandomForestClassifier(random_state=config.RANDOM_STATE)),
        ]),
        {"clf__n_estimators": [200, 500], "clf__max_depth": [3, 5, None]},
    ),
    "SVM": (
        Pipeline([
            ("var", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=config.RANDOM_STATE)),
        ]),
        {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", "auto"]},
    ),
}


def load_merged():
    clinical = pd.read_csv(config.CLINICAL_CLEAN_CSV)
    features = pd.read_csv(config.FEATURES_CSV)
    if "roi_used" in features.columns:
        features = features.drop(columns=["roi_used"])

    merged = clinical.merge(features, on="PatientID", how="inner")
    merged.to_csv(config.MERGED_CSV, index=False)
    print(f"Merged clinical + radiomics: {merged.shape[0]} patients "
          f"(clinical had {len(clinical)}, radiomics had {len(features)})")

    y = merged["Outcome"].astype(int).values
    clinical_cols = [c for c in clinical.columns if c not in ("PatientID", "Outcome")]
    radiomics_cols = [c for c in features.columns if c != "PatientID"]

    X_clin = merged[clinical_cols].values
    X_rad = merged[radiomics_cols].values
    X_comb = merged[clinical_cols + radiomics_cols].values
    return {"clinical": X_clin, "radiomics": X_rad, "combined": X_comb}, y


def nested_cv_auc(X, y, model_name):
    pipe, grid = MODEL_GRIDS[model_name]
    outer = StratifiedKFold(n_splits=config.OUTER_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_aucs = []
    for train_idx, test_idx in outer.split(X, y):
        inner = StratifiedKFold(n_splits=config.INNER_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1)
        search.fit(X[train_idx], y[train_idx])
        proba = search.predict_proba(X[test_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y[test_idx], proba))
    return np.array(fold_aucs)


def permutation_test(X, y, model_name, observed_auc, n_perm):
    pipe, grid = MODEL_GRIDS[model_name]
    perm_aucs = []
    for i in range(n_perm):
        y_perm = shuffle(y, random_state=i)
        outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=i)
        aucs = []
        for train_idx, test_idx in outer.split(X, y_perm):
            pipe.fit(X[train_idx], y_perm[train_idx])
            proba = pipe.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(y_perm[test_idx], proba))
        perm_aucs.append(np.mean(aucs))
    perm_aucs = np.array(perm_aucs)
    return (np.sum(perm_aucs >= observed_auc) + 1) / (n_perm + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=config.N_PERMUTATIONS)
    args = parser.parse_args()

    if not config.CLINICAL_CLEAN_CSV.exists() or not config.FEATURES_CSV.exists():
        print("ERROR: run 01_extract_radiomics.py and 02_prepare_clinical.py first.")
        sys.exit(1)

    blocks, y = load_merged()
    print(f"Class balance — metastasis: {y.mean():.1%} ({y.sum()}/{len(y)})")
    if len(y) < 30:
        print("NOTE: small cohort — AUC estimates will have wide confidence intervals. "
              "Treat results as directional, not definitive.")

    results = []
    for block_name, X in blocks.items():
        for model_name in MODEL_GRIDS:
            print(f"\n[{block_name} | {model_name}] running nested CV...")
            aucs = nested_cv_auc(X, y, model_name)
            mean_auc, std_auc = aucs.mean(), aucs.std()
            print(f"  AUC = {mean_auc:.3f} +/- {std_auc:.3f}")

            print(f"  running permutation test ({args.n_perm} permutations)...")
            p_val = permutation_test(X, y, model_name, mean_auc, args.n_perm)
            print(f"  permutation p-value = {p_val:.4f}")

            results.append({
                "feature_block": block_name, "model": model_name,
                "mean_auc": mean_auc, "std_auc": std_auc,
                "fold_aucs": aucs.tolist(), "permutation_p": p_val,
                "n_features": X.shape[1], "n_patients": X.shape[0],
            })

    results_df = pd.DataFrame(results).sort_values(["feature_block", "mean_auc"], ascending=[True, False])
    results_df.to_csv(config.RESULTS_CSV, index=False)
    print(f"\nSaved comparison table to {config.RESULTS_CSV}")
    print("\n=== Summary (best model per block) ===")
    print(results_df.loc[results_df.groupby("feature_block")["mean_auc"].idxmax()]
          [["feature_block", "model", "mean_auc", "std_auc", "permutation_p"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
