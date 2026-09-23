"""
For each patient: read the CT series, PT (PET) series, and RTSTRUCT contour,
build a binary tumor mask aligned to the CT grid, resample it onto the PET
grid too, then run PyRadiomics on both CT and PET.

Requires: SimpleITK, pydicom, rt_utils, pyradiomics
    pip install SimpleITK pydicom rt_utils pyradiomics

Usage:
    python scripts/01_extract_radiomics.py
"""
import sys
import traceback
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
from rt_utils import RTStructBuilder
from radiomics import featureextractor

logging.getLogger("radiomics").setLevel(logging.ERROR)


def build_extractor(bin_width=None, bin_count=None):
    settings = {
        "resampledPixelSpacing": list(config.RESAMPLE_SPACING),
        "interpolator": sitk.sitkBSpline,
        "normalize": False,
        "preCrop": True,
        "padDistance": 10,
    }
    if bin_count is not None:
        settings["binCount"] = bin_count
    else:
        settings["binWidth"] = bin_width
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    if config.ENABLE_WAVELET:
        extractor.enableImageTypeByName("Wavelet")
    if config.ENABLE_LOG:
        extractor.enableImageTypeByName("LoG")
    extractor.enableAllFeatures()
    return extractor

def read_series_as_image(dicom_dir):
    reader = sitk.ImageSeriesReader()
    file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    reader.SetFileNames(file_names)
    return reader.Execute()


def series_uid_of_dir(dicom_dir):
    """Read SeriesInstanceUID from the first .dcm file in a folder."""
    dcm_files = list(Path(dicom_dir).glob("*.dcm"))
    if not dcm_files:
        return None
    ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
    return getattr(ds, "SeriesInstanceUID", None)


def rtstruct_referenced_series_uid(rt_path):
    """Pull the SeriesInstanceUID an RTSTRUCT file's contours are drawn on."""
    ds = pydicom.dcmread(rt_path, stop_before_pixels=True)
    try:
        return (ds.ReferencedFrameOfReferenceSequence[0]
                  .RTReferencedStudySequence[0]
                  .RTReferencedSeriesSequence[0]
                  .SeriesInstanceUID)
    except Exception:
        pass
    try:
        roi_contour = ds.ROIContourSequence[0]
        contour = roi_contour.ContourSequence[0]
        ref_sop_uid = contour.ContourImageSequence[0].ReferencedSOPInstanceUID
        return ("SOP", ref_sop_uid)
    except Exception:
        return None


def find_roi_mask(rtstruct_dir, ct_dir):
    """Return (mask_array_zyx, roi_name) or (None, None) if no matching ROI."""
    dcm_files = list(Path(rtstruct_dir).glob("*.dcm"))
    if not dcm_files:
        return None, None
    rt_path = dcm_files[0]

    try:
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(ct_dir), rt_struct_path=str(rt_path)
        )
    except Exception as e:
        print(f"    [warn] rt_utils could not load RTSTRUCT: {e}")
        return None, None

    roi_names = rtstruct.get_roi_names()
    match = next((n for n in roi_names if n in config.ROI_NAME_CANDIDATES), None)
    if match is None:
        match = roi_names[0] if roi_names else None
    if match is None:
        return None, None

    mask_xyz = rtstruct.get_roi_mask_by_name(match)  # shape (rows, cols, slices)
    mask_zyx = np.transpose(mask_xyz, (2, 0, 1)).astype(np.uint8)  # -> (slices, rows, cols)
    return mask_zyx, match


def mask_to_image(mask_zyx, reference_image):
    mask_img = sitk.GetImageFromArray(mask_zyx)
    mask_img.CopyInformation(reference_image)
    return mask_img


def resample_mask_to(mask_img, target_image):
    return sitk.Resample(mask_img, target_image, sitk.Transform(),
                          sitk.sitkNearestNeighbor, 0, mask_img.GetPixelID())


def main():
    if not config.SERIES_INDEX_CSV.exists():
        print("ERROR: run 00_scan_dicom.py first.")
        sys.exit(1)

    idx = pd.read_csv(config.SERIES_INDEX_CSV)
    ct_extractor = build_extractor(bin_width=config.CT_BIN_WIDTH)
    pet_extractor = build_extractor(bin_count=config.PET_BIN_COUNT)
    rows = []
    patients = sorted(idx["PatientID"].unique())
    print(f"Processing {len(patients)} patients...")

    for i, pid in enumerate(patients, 1):
        print(f"[{i}/{len(patients)}] {pid}")
        p_rows = idx[idx["PatientID"] == pid]

        ct_rows = p_rows[p_rows["Modality"] == "CT"]
        pt_rows = p_rows[p_rows["Modality"] == "PT"]
        rt_rows = p_rows[p_rows["Modality"] == "RTSTRUCT"]

        if ct_rows.empty or rt_rows.empty:
            print(f"    [skip] missing CT or RTSTRUCT for {pid}")
            continue

        # This dataset has multiple RTSTRUCT files per patient, each drawn on a
        # different reference series (CT, PET, or various MR series). We need
        # the one drawn on CT, so we check each RTSTRUCT's referenced series
        # UID against each candidate CT series until we find a match.
        rt_dir = None
        ct_dir = None
        for _, ct_r in ct_rows.iterrows():
            candidate_ct_dir = ct_r["dir"]
            candidate_ct_uid = series_uid_of_dir(candidate_ct_dir)
            for _, rt_r in rt_rows.iterrows():
                candidate_rt_dir = rt_r["dir"]
                rt_file = next(Path(candidate_rt_dir).glob("*.dcm"), None)
                if rt_file is None:
                    continue
                ref_uid = rtstruct_referenced_series_uid(rt_file)
                if isinstance(ref_uid, tuple):
                    continue
                if ref_uid == candidate_ct_uid:
                    rt_dir = candidate_rt_dir
                    ct_dir = candidate_ct_dir
                    break
            if rt_dir is not None:
                break

        if rt_dir is None or ct_dir is None:
            print(f"    [skip] no RTSTRUCT found that references any CT series for {pid} "
                  f"({len(ct_rows)} CT series, {len(rt_rows)} RTSTRUCT files checked)")
            continue

        try:
            ct_img = read_series_as_image(ct_dir)
        except Exception as e:
            print(f"    [skip] could not read CT series: {e}")
            continue

        mask_zyx, roi_name = find_roi_mask(rt_dir, ct_dir)
        if mask_zyx is None:
            print(f"    [skip] no usable ROI found for {pid}")
            continue
        if mask_zyx.shape != sitk.GetArrayFromImage(ct_img).shape:
            print(f"    [skip] mask/CT shape mismatch for {pid}: "
                  f"{mask_zyx.shape} vs {sitk.GetArrayFromImage(ct_img).shape}")
            continue

        ct_mask_img = mask_to_image(mask_zyx, ct_img)

        row = {"PatientID": pid, "roi_used": roi_name}
        try:
            ct_feats = ct_extractor.execute(ct_img, ct_mask_img, label=1)
            for k, v in ct_feats.items():
                if not k.startswith("diagnostics_"):
                    row[f"CT_{k}"] = float(v)
        except Exception:
            print(f"    [warn] CT extraction failed for {pid}:")
            print(traceback.format_exc())

        if not pt_rows.empty:
            pt_dir = pt_rows.iloc[0]["dir"]
            try:
                pt_img = read_series_as_image(pt_dir)
                pt_mask_img = resample_mask_to(ct_mask_img, pt_img)
                pet_feats = pet_extractor.execute(pt_img, pt_mask_img, label=1)
                for k, v in pet_feats.items():
                    if not k.startswith("diagnostics_"):
                        row[f"PET_{k}"] = float(v)
            except Exception:
                print(f"    [warn] PET extraction failed for {pid}:")
                print(traceback.format_exc())

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(config.FEATURES_CSV, index=False)
    print(f"\nSaved {df.shape[0]} patients x {df.shape[1]-2} features to {config.FEATURES_CSV}")


if __name__ == "__main__":
    main()