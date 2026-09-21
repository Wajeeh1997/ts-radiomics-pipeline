"""
Clean the INFOclinical_STS.xlsx clinical/outcome spreadsheet.

The exact column names in this file haven't been verified against your actual
download yet — this script prints all columns on a mismatch so you can fix
config.PATIENT_ID_COL / config.OUTCOME_COL in one place rather than guessing
blind.

Usage:
    python scripts/02_prepare_clinical.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import pandas as pd
import numpy as np


def main():
    if not config.CLINICAL_XLSX.exists():
        print(f"ERROR: {config.CLINICAL_XLSX} not found.")
        print("Download INFOclinical_STS.xlsx from https://wiki.cancerimagingarchive.net/x/ZYBEAQ "
              "(under 'Clinical Data') and place it at data/INFOclinical_STS.xlsx")
        sys.exit(1)

    df = pd.read_excel(config.CLINICAL_XLSX)
    df.columns = [str(c).strip() for c in df.columns]

    print(f"Loaded clinical file: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    if config.PATIENT_ID_COL not in df.columns:
        print(f"\nERROR: config.PATIENT_ID_COL = '{config.PATIENT_ID_COL}' not found.")
        print("Set config.PATIENT_ID_COL to the correct column name from the list above and rerun.")
        sys.exit(1)

    if config.OUTCOME_COL not in df.columns:
        print(f"\nERROR: config.OUTCOME_COL = '{config.OUTCOME_COL}' not found.")
        print("Set config.OUTCOME_COL to the correct column name from the list above and rerun.")
        sys.exit(1)

    df = df.rename(columns={config.PATIENT_ID_COL: "PatientID"})

    # Outcome may be coded as text (e.g. "yes"/"no", "M+"/"M-") or 0/1 — normalize.
    outcome = df[config.OUTCOME_COL]
    if outcome.dtype == object:
        outcome = outcome.astype(str).str.strip().str.lower()
        positive_tokens = {"yes", "y", "1", "true", "m+", "positive", "pos"}
        outcome = outcome.isin(positive_tokens).astype(int)
    else:
        outcome = outcome.fillna(0).astype(int)
    df["Outcome"] = outcome

    print(f"\nOutcome distribution: {df['Outcome'].value_counts().to_dict()}")
    if df["Outcome"].nunique() < 2:
        print("WARNING: outcome column has only one class after encoding — "
              "double-check config.OUTCOME_COL and the positive_tokens mapping above.")

    # Keep remaining columns as clinical covariates; one-hot encode categoricals,
    # median-impute numerics.
    covariate_cols = [c for c in df.columns if c not in ("PatientID", config.OUTCOME_COL, "Outcome")]
    clean = df[["PatientID", "Outcome"] + covariate_cols].copy()

    for col in covariate_cols:
        if clean[col].dtype == object:
            clean[col] = clean[col].astype("category")
    cat_cols = [c for c in covariate_cols if clean[c].dtype.name == "category"]
    clean = pd.get_dummies(clean, columns=cat_cols, dummy_na=True)

    numeric_cols = clean.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ("PatientID", "Outcome")]
    for col in numeric_cols:
        clean[col] = clean[col].fillna(clean[col].median())

    clean.to_csv(config.CLINICAL_CLEAN_CSV, index=False)
    print(f"\nSaved cleaned clinical table ({clean.shape[0]} x {clean.shape[1]}) "
          f"to {config.CLINICAL_CLEAN_CSV}")


if __name__ == "__main__":
    main()
