# 🩻 STS Radiomics vs. Clinical Variables: Predicting Lung Metastasis in Soft-Tissue Sarcoma

> **Do PET/CT radiomics features actually add predictive value beyond routine clinical variables?**

A reproducible **PET/CT radiomics + machine learning pipeline** investigating whether image-derived radiomics features improve prediction of lung metastasis in soft-tissue sarcoma (STS), or whether clinical variables alone provide sufficient predictive information.

This project reproduces the methodology of a previous **clinical-vs-radiomics study of hematological toxicity after ¹⁷⁷Lu-PSMA therapy**, applying the same general framework to an **independent, publicly available cohort** with a different outcome and imaging protocol.

The objective is not to assume that radiomics improves prediction, but to test whether it actually provides **incremental predictive value over clinical information**.

---

## 🔬 Key Result

### Clinical variables performed best

| Feature block           | Best model    |  Mean AUC | Std AUC | Permutation *p* | Features | Patients |
| ----------------------- | ------------- | --------: | ------: | --------------: | -------: | -------: |
| 🩺 **Clinical only**    | Random Forest | **0.835** |   0.116 |           0.002 |       72 |       51 |
| 🔬 Clinical + Radiomics | LASSO         |     0.808 |   0.150 |           0.002 |      286 |       51 |
| 🩻 Radiomics only       | Random Forest |     0.789 |   0.154 |           0.003 |      214 |       51 |

**Clinical variables alone achieved the highest mean AUC (0.835).**

All three feature blocks performed significantly better than chance under a **1,000-permutation test**. However, adding radiomics features did not improve performance over clinical variables alone and modestly reduced the observed mean AUC in this cohort.

This provides an independent test of the central pattern observed in the thesis work on which this pipeline is based: **clinical information can outperform high-dimensional radiomics features when evaluated under rigorous cross-validation.**

---

## 🧬 Radiomics Feature Stability

Why didn't the large radiomics feature set improve performance?

A **5-fold nested cross-validation + LASSO feature-selection stability analysis** provides one possible explanation:

* **172 / 214 radiomics features (80%) were never selected in any fold**
* Only **3 radiomics features were selected consistently across all 5 folds**
* Most selected features were therefore **fold-specific rather than consistently reproducible**

This suggests that the radiomics signal captured by the models may be relatively unstable in this small cohort.

> ⚠️ **Important:** This is a methodological finding from a **51-patient cohort**, not evidence that radiomics is broadly ineffective for soft-tissue sarcoma or metastasis prediction.

---

## 🔎 Exploratory Findings

### Age distribution

Patients who developed lung metastasis tended to cluster roughly around **55–70 years of age**, compared with the broader approximately **15–85-year age distribution** among patients without recorded lung metastasis.

This is consistent with `Age` appearing among the relatively stable features in the combined-block feature-selection analysis.

### Radiomics feature redundancy

Shape-derived features such as:

* Elongation
* Flatness
* Major/minor axis lengths
* Mesh volume
* Surface area

show strong inter-correlation, as expected from their shared geometric basis.

This redundancy is one plausible contributor to the large proportion of radiomics features that were never selected by LASSO across folds: many features encode **overlapping rather than independent information**.

---

# 🎯 Research Question

The project asks a deliberately simple question:

> **When routine clinical information is already available, does adding PET/CT radiomics actually improve prediction of lung metastasis?**

Three feature strategies are compared:

1. 🩺 **Clinical variables only**
2. 🩻 **Radiomics features only**
3. 🔬 **Clinical + radiomics features**

The analysis evaluates whether high-dimensional imaging biomarkers provide **incremental predictive value** beyond a conventional clinical baseline.

---

# 📊 Dataset

### Source

**TCIA Soft-Tissue-Sarcoma Collection**
Vallières et al., accessed through **IDC (Imaging Data Commons)**.

### Cohort

* **51 / 62 patients** with usable CT, PET, RTSTRUCT tumor contours, and recorded outcome
* **11 patients excluded** because of missing follow-up
* **19 positive** lung-metastasis cases
* **32 negative** cases

### Outcome

Binary lung-metastasis status derived from the:

```text
Outcome (recurrence, mets)
```

clinical variable.

### Imaging

* CT
* ¹⁸F-FDG PET
* RTSTRUCT tumor contours
* `GTV_Mass` ROI

The imaging data are publicly available through TCIA/IDC but are **not included in this repository** because of dataset licensing restrictions and storage requirements.

---

# 🧠 Methodology

The complete pipeline consists of seven stages.

### 1. DICOM indexing

**`00_scan_dicom.py`**

Indexes DICOM series for each patient by modality without reading full image volumes.

### 2. Clinical preprocessing

**`02_prepare_clinical.py`**

* Cleans the clinical spreadsheet
* Derives the binary lung-metastasis outcome
* Removes outcome-adjacent variables
* One-hot encodes categorical variables
* Median-imputes numerical variables

### 3. Radiomics extraction

**`01_extract_radiomics.py`**

For each patient:

1. Identifies the appropriate CT series
2. Matches the RTSTRUCT to its referenced CT series
3. Builds a 3D tumor mask
4. Crops the image around the tumor
5. Resamples to **1 mm isotropic spacing**
6. Extracts PyRadiomics features from CT and PET

Extracted feature families include:

* First-order statistics
* Shape features
* Texture features

### 4. Exploratory data analysis

**`eda.py`**

Performs:

* Outcome-balance analysis
* Merge-coverage checks
* Missingness analysis
* Near-zero-variance detection
* Feature correlation analysis

### 5. Machine-learning evaluation

**`03_ml_pipeline.py`**

Uses **nested cross-validation**:

* 5 outer folds
* 4 inner folds

and compares:

* LASSO
* Random Forest
* SVM

across:

* Clinical features
* Radiomics features
* Combined clinical + radiomics features

Each block/model combination is additionally evaluated using **1,000 permutation tests**.

### 6. Feature-selection stability

**`05_feature_stability.py`**

Tracks LASSO-selected features across the outer folds to distinguish:

**stable signal**

from

**fold-specific feature selection.**

### 7. Results generation

**`04_report.py`**

Generates the final model-comparison table and summary statistics.

---

# ⚙️ Notable Implementation Details

## 1. RTSTRUCT → CT Series Matching

This dataset contains up to **five RTSTRUCT files per patient**, referencing different imaging series such as:

* CT
* PET
* T1
* T2-FS
* Aligned variants

Rather than assuming that the first RTSTRUCT encountered is the correct one, the pipeline reads the RTSTRUCT's:

```text
ReferencedSeriesInstanceUID
```

and matches it against candidate CT series using the corresponding UID.

Without this matching step:

> **45 of 51 patients failed**

with:

```text
RTStruct references image(s) not contained in input series data
```

Correct RTSTRUCT-to-series matching was therefore essential for reliable radiomics extraction.

---

## 2. PET Discretization: Bin Count Instead of Bin Width

The PET images in this IDC distribution contain **raw scanner intensity values rather than SUV-normalized values**.

Within the tumor mask, values can span approximately:

```text
0 – 63,000
```

Using a conventional fixed `binWidth` would create an extremely large number of gray levels. For PET, this can make GLCM construction prohibitively memory-intensive — particularly on an 8 GB machine.

Instead, the pipeline uses:

```python
binCount = 64
```

This provides a fixed number of intensity levels and is consistent with the general IBSI approach for intensity-unnormalized images.

---

## 3. Pre-Cropping Before Resampling

Large PET/CT volumes can consume substantial memory.

Instead of:

```text
Whole image → resample → crop
```

the pipeline performs:

```text
Whole image → crop around tumor → resample
```

The crop uses approximately **10 mm of padding around the tumor bounding box**.

This means memory requirements scale more closely with the tumor region rather than the entire whole-body or whole-limb scan.

---

# 🚨 Data Leakage Fix

An earlier version of the clinical feature set contained:

```text
Time – diagnosis to outcome (days)
Status (NED, AWD, D)
Time – diagnosis to last follow-up (days)
```

These variables are observed after or in connection with the outcome event and therefore introduce **outcome leakage**.

Including them inflated the clinical-only AUC from:

### `0.835 → 0.917`

The 0.917 result was therefore considered implausible and indicative of leakage.

These variables are now explicitly excluded through:

```python
LEAKY_COLS
```

in:

```text
02_prepare_clinical.py
```

The reported **0.835 clinical AUC** is based on the corrected feature set.

This correction is particularly important because it demonstrates why **leakage prevention is critical in small medical-imaging datasets**, where a small number of inappropriate variables can dramatically inflate apparent model performance.

---

# 🛠️ Setup

Install the standard dependencies:

```bash
pip install -r requirements.txt
```

For Windows, a Conda environment is recommended because PyRadiomics may not provide a convenient prebuilt wheel for recent Python versions without a C++ compiler.

```bash
conda create -n sts python=3.9 -y
conda activate sts

conda install -c radiomics -c conda-forge \
    pyradiomics simpleitk pydicom pandas scikit-learn \
    matplotlib openpyxl -y

pip install rt_utils idc-index tabulate
```

---

# 📥 Data

The imaging data are **not included in this repository**.

Download the data separately.

### Imaging

Use `idc-index` to retrieve the TCIA Soft-Tissue-Sarcoma collection:

```python
collection_id = "soft_tissue_sarcoma"
```

### Clinical data

Download:

```text
INFOclinical_STS.xlsx
```

from the TCIA Soft-Tissue-Sarcoma collection under **Clinical Data**.

Place the data under:

```text
data/
```

next to the scripts.

Alternatively, set:

```text
STS_DATA_DIR
```

to another location.

A shorter path can be useful on Windows because IDC's nested directory structure can otherwise approach the traditional **260-character path limit**.

---

# ▶️ Usage

Run the pipeline in the following order:

```bash
python scripts/00_scan_dicom.py
python scripts/02_prepare_clinical.py
python scripts/01_extract_radiomics.py
python scripts/eda.py
python scripts/03_ml_pipeline.py
python scripts/05_feature_stability.py
python scripts/04_report.py
```

---

# ⚠️ Limitations

### Small cohort

The analysis contains only:

* **51 patients**
* **19 metastatic events**

AUC variability is therefore substantial, with standard deviations ranging from approximately **0.12–0.15**.

The results should be interpreted as **directional methodological findings**, not as a validated clinical prediction model.

### No external validation

The analysis uses a single publicly available cohort and does not include an independent external validation cohort.

### PET is not SUV-normalized

PET radiomics are derived from raw scanner intensity values using fixed-bin-count discretization.

Therefore, the resulting PET radiomics values should **not be interpreted as directly comparable to SUV-based radiomics measurements** reported in other studies.

### Single dataset / protocol

The analysis does not establish generalizability across:

* Institutions
* Scanners
* Acquisition protocols
* Patient populations

### Feature stability

Feature stability was assessed using **LASSO** and was not independently evaluated across all three model families.

---

# 📌 Takeaway

This project tests a fundamental question in medical-imaging machine learning:

> **Does adding hundreds of radiomics features actually improve prediction when clinical information is already available?**

In this **51-patient soft-tissue-sarcoma cohort**, clinical variables achieved the highest observed mean AUC:

### **Clinical only: 0.835**

compared with:

* **Clinical + radiomics: 0.808**
* **Radiomics only: 0.789**

The feature-stability analysis provides a possible explanation: **80% of radiomics features were never selected across the nested CV folds**, while only three features were selected consistently.

The result should not be interpreted as evidence that radiomics is inherently ineffective.

Instead, it highlights a broader methodological lesson:

> **More features do not automatically mean more predictive information.**

For small medical-imaging datasets, **careful preprocessing, leakage prevention, nested validation, feature stability, and reproducibility are critical when evaluating whether complex imaging biomarkers provide genuine incremental value.**

---

# 📚 References & Data Sources

* **The Cancer Imaging Archive (TCIA)** — Soft-Tissue-Sarcoma Collection
* **Imaging Data Commons (IDC)**
* **PyRadiomics**
* **Image Biomarker Standardisation Initiative (IBSI)**
* Vallières et al. — Soft-Tissue-Sarcoma PET/CT dataset and associated clinical data

---

# 🙏 Acknowledgments

The methodology was informed by prior radiomics-vs-clinical work using a different cohort, imaging modality pairing, and clinical outcome.

This project was built entirely using the **public TCIA Soft-Tissue-Sarcoma collection**. No private or institutional patient data are included in this repository.
