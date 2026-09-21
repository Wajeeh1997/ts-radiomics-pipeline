"""
Central configuration for the Soft-Tissue-Sarcoma radiomics-vs-clinical pipeline.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Root of the raw DICOM download from idc-index (contains CT/PT/MR/RTSTRUCT series
# nested in whatever hierarchy idc-index used — 01_scan_dicom.py finds them by
# reading DICOM tags directly, so the exact folder layout doesn't matter).
RAW_DICOM_ROOT = PROJECT_ROOT / "data" / "soft_tissue_sarcoma"

# Clinical spreadsheet downloaded separately from TCIA (not included in the IDC
# imaging download) — https://wiki.cancerimagingarchive.net/x/ZYBEAQ
CLINICAL_XLSX = PROJECT_ROOT / "data" / "INFOclinical_STS.xlsx"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CONVERTED_DIR = OUTPUT_DIR / "converted"          # per-patient NIfTI CT/PT + mask
SERIES_INDEX_CSV = OUTPUT_DIR / "series_index.csv"
FEATURES_CSV = OUTPUT_DIR / "radiomics_features.csv"
CLINICAL_CLEAN_CSV = OUTPUT_DIR / "clinical_clean.csv"
MERGED_CSV = OUTPUT_DIR / "merged_dataset.csv"
RESULTS_CSV = OUTPUT_DIR / "model_comparison_results.csv"

# ---- Image discretization (PyRadiomics) ------------------------------------
CT_BIN_WIDTH = 25       # Hounsfield units
PET_BIN_WIDTH = 0.25    # SUV units
RESAMPLE_SPACING = (1.0, 1.0, 1.0)
ENABLE_WAVELET = False
ENABLE_LOG = False

# RTSTRUCT ROI name matching — the Vallières et al. STS dataset's tumor contour
# ROI name varies by patient in practice; 01_scan_dicom.py will print every ROI
# name it finds on the first run so you can confirm/adjust this list.
ROI_NAME_CANDIDATES = ["GTV", "GTV_MASS", "GTV1", "Tumor", "Mass"]

# ---- Clinical / outcome -----------------------------------------------------
# Column names in INFOclinical_STS.xlsx — verify against the actual file on
# first run (02_prepare_clinical.py will print all columns if these aren't found).
PATIENT_ID_COL = "Patient_ID"
OUTCOME_COL = "Metastases"    # lung metastasis status — confirm exact column name

# ---- Modeling ----------------------------------------------------------------
RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 4   # smaller inner fold count than the HECKTOR version — this cohort is only 51 patients
N_PERMUTATIONS = 1000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
