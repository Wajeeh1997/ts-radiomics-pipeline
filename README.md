# Soft-Tissue-Sarcoma Radiomics vs. Clinical Prediction

End-to-end PET/CT radiomics-vs-clinical pipeline, reproducing the structure of
the thesis's toxicity-prediction methodology (bone-marrow PET/CT radiomics vs.
clinical variables, LASSO/RF/SVM, nested CV, permutation testing) on a public
cohort: the **TCIA/IDC Soft-Tissue-Sarcoma** dataset (Vallières et al. 2015,
*Phys Med Biol*) — 51 patients, FDG-PET/CT + MRI, with a real clinical outcome
(19/51 patients developed lung metastases during follow-up).

## 1. Environment

```bash
python -m venv sts_env
# Windows:  sts_env\Scripts\python.exe -m pip install -r requirements.txt
# macOS/Linux:
source sts_env/bin/activate
pip install -r requirements.txt
```

## 2. Data

**Imaging** (CT, PET, MR, RTSTRUCT tumor contours — ~9.6 GB), no account needed:
```python
from idc_index import IDCClient
IDCClient.client().download_from_selection(
    collection_id="soft_tissue_sarcoma", downloadDir="./data/soft_tissue_sarcoma"
)
```

**Clinical/outcome spreadsheet** (`INFOclinical_STS.xlsx`, ~12 KB) — not bundled
with the IDC imaging download, grab it separately from
https://wiki.cancerimagingarchive.net/x/ZYBEAQ (linked under "Clinical Data")
and place it at `data/INFOclinical_STS.xlsx`.

## 3. Run order

```bash
python scripts/00_scan_dicom.py         # index all DICOM series + list ROI names found
python scripts/01_extract_radiomics.py  # PyRadiomics on CT + PET, per patient
python scripts/02_prepare_clinical.py   # clean clinical vars, binary lung-met outcome
python scripts/03_ml_pipeline.py        # nested-CV: clinical vs radiomics vs combined
python scripts/04_report.py             # comparison figure + summary table
```

**Before step 1**, check the ROI names `00_scan_dicom.py` prints against
`config.ROI_NAME_CANDIDATES` — this dataset's tumor contour naming varies
slightly by patient, and the script needs to know which ROI is the tumor.

**Before step 3**, check `02_prepare_clinical.py`'s printed column list against
`config.PATIENT_ID_COL` / `config.OUTCOME_COL` — verify these match the actual
downloaded spreadsheet before trusting the encoded outcome.

## What's genuinely different from the thesis
Different disease (soft-tissue sarcoma vs. prostate cancer), different outcome
(lung metastasis vs. hematological toxicity), much smaller cohort (51 vs. 59,
but far less statistical power once split across folds). This validates the
*same methodology* on an independent public cohort — it is not a replication
of the original clinical finding. Say so explicitly if you write this up
anywhere.

## Publishing this to GitHub

This repo is meant to hold **code only** — the `.gitignore` already excludes
`data/` (patient imaging) and `outputs/converted/` (derived NIfTI volumes) so
you don't accidentally push gigabytes of medical images. `outputs/*.csv` and
the summary figure are small and fine to include if you want the results
visible in the repo.

```bash
cd sts_radiomics
git init
git add .
git commit -m "Initial commit: STS radiomics-vs-clinical pipeline"
```

Then create an empty repo on GitHub (via the website, or `gh repo create` if
you have the GitHub CLI installed), and push:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

If you're prompted for a password and it fails — GitHub retired password auth
years ago — use a Personal Access Token instead (GitHub → Settings →
Developer settings → Personal access tokens) as the password when Git asks.

## License note
The Soft-Tissue-Sarcoma dataset itself is CC BY 3.0 (attribution required if
you publish results — cite Vallières et al. 2015 and the TCIA data citation).
This code is yours to license however you like; an MIT `LICENSE` file is
included as a permissive default — swap it if you'd prefer something else.
