"""
Scan the raw DICOM download and build an index of every series, grouped by
PatientID + Modality + SeriesInstanceUID. Reads DICOM tags directly instead of
trusting folder structure, so it works regardless of how idc-index laid the
files out on disk.

Usage:
    python scripts/00_scan_dicom.py
"""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import pydicom
import pandas as pd


def main():
    if not config.RAW_DICOM_ROOT.exists():
        print(f"ERROR: {config.RAW_DICOM_ROOT} not found. Point config.RAW_DICOM_ROOT "
              "at your idc-index download folder.")
        sys.exit(1)

    print(f"Scanning {config.RAW_DICOM_ROOT} for DICOM files (this reads one file "
          "per series, not every slice, so it's quick)...")

    series_files = defaultdict(list)   # (PatientID, SeriesInstanceUID) -> [file paths]
    series_meta = {}                   # (PatientID, SeriesInstanceUID) -> dict of tags
    roi_names_seen = set()

    dcm_files = list(config.RAW_DICOM_ROOT.rglob("*.dcm"))
    print(f"Found {len(dcm_files)} .dcm files on disk.")

    for i, fp in enumerate(dcm_files):
        if i % 2000 == 0:
            print(f"  ...{i}/{len(dcm_files)}")
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True)
        except Exception:
            continue

        pid = str(getattr(ds, "PatientID", "UNKNOWN"))
        series_uid = str(getattr(ds, "SeriesInstanceUID", "UNKNOWN"))
        modality = str(getattr(ds, "Modality", "UNKNOWN"))
        key = (pid, series_uid)
        series_files[key].append(fp)

        if key not in series_meta:
            series_meta[key] = {
                "PatientID": pid,
                "SeriesInstanceUID": series_uid,
                "Modality": modality,
                "StudyInstanceUID": str(getattr(ds, "StudyInstanceUID", "")),
            }

        if modality == "RTSTRUCT":
            try:
                ds_full = pydicom.dcmread(fp)
                for roi in ds_full.StructureSetROISequence:
                    roi_names_seen.add(str(roi.ROIName))
            except Exception as e:
                print(f"  [warn] could not read ROI names from {fp}: {e}")

    rows = []
    for key, meta in series_meta.items():
        meta = dict(meta)
        meta["n_files"] = len(series_files[key])
        meta["dir"] = str(series_files[key][0].parent)
        rows.append(meta)

    df = pd.DataFrame(rows)
    df.to_csv(config.SERIES_INDEX_CSV, index=False)

    print(f"\nSaved series index ({df.shape[0]} series) to {config.SERIES_INDEX_CSV}")
    print("\nModality counts:")
    print(df.groupby("Modality")["PatientID"].nunique())
    print(f"\nUnique patients: {df['PatientID'].nunique()}")

    print("\nROI names found across all RTSTRUCT files:")
    for name in sorted(roi_names_seen):
        print(f"  - {name}")
    print("\nCompare this list against config.ROI_NAME_CANDIDATES and adjust if needed "
          "before running 01_extract_radiomics.py.")


if __name__ == "__main__":
    main()
