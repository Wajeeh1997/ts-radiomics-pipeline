"""
Central configuration for the Soft-Tissue-Sarcoma radiomics-vs-clinical pipeline.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Folder holding the downloaded data. Defaults to ./data next to this file;
# set the STS_DATA_DIR environment variable to use a different location.
DATA_DIR = Path(os.environ.get("STS_DATA_DIR", PROJECT_ROOT / "data"))

# Root of the raw DICOM download from idc-index (contains CT/PT/MR/RTSTRUCT series
# nested in whatever hierarchy idc-index used — 01_scan_dicom.py finds them by
# reading DICOM tags directly, so the exact folder layout doesn't matter).
RAW_DICOM_ROOT = DATA_DIR / "soft_tissue_sarcoma"

# Clinical spreadsheet downloaded separately from TCIA (not included in the IDC
# imaging download) — https://wiki.cancerimagingarchive.net/x/ZYBEAQ
CLINICAL_XLSX = DATA_DIR / "INFOclinical_STS.xlsx"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CONVERTED_DIR = OUTPUT_DIR / "converted"          # per-patient NIfTI CT/PT + mask
SERIES_INDEX_CSV = OUTPUT_DIR / "series_index.csv"
FEATURES_CSV = OUTPUT_DIR / "radiomics_features.csv"
CLINICAL_CLEAN_CSV = OUTPUT_DIR / "clinical_clean.csv"
MERGED_CSV = OUTPUT_DIR / "merged_dataset.csv"
RESULTS_CSV = OUTPUT_DIR / "model_comparison_results.csv"

# ---- Image discretization (PyRadiomics) ------------------------------------
CT_BIN_WIDTH = 25       # Hounsfield units
PET_BIN_COUNT = 64
RESAMPLE_SPACING = (1.0, 1.0, 1.0)
ENABLE_WAVELET = False
ENABLE_LOG = False

# RTSTRUCT ROI name matching — confirmed against this download via 00_scan_dicom.py:
# ROI names present are GTV_Edema, GTV_Mass, GTV_Res+edema, GTV_Research.
# GTV_Mass is the tumor contour (the others are edema / research-only contours).
ROI_NAME_CANDIDATES = ["GTV_Mass"]

# ---- Clinical / outcome -----------------------------------------------------
# Column names in INFOclinical_STS.xlsx — confirmed against the actual file.
PATIENT_ID_COL = "Patient ID"
OUTCOME_COL = "Outcome (recurrence, mets)"

# ---- Modeling ----------------------------------------------------------------
RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 4   # smaller inner fold count than the HECKTOR version — this cohort is only 51 patients
N_PERMUTATIONS = 100

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONVERTED_DIR.mkdir(parents=True, exist_ok=True)