"""
Builds a "representative case" figure for the paper: one patient's CT slice,
the corresponding PET slice overlaid in color, and the RTSTRUCT-derived tumor
contour outline drawn on top — the kind of example-case image common in
radiomics papers (Fig. X placeholder in paper_ieeetran.tex / paper_preview.tex).

This reuses the same series-matching and mask-building logic as
01_extract_radiomics.py, so it needs series_index.csv to already exist
(run 00_scan_dicom.py first if it doesn't).

Usage:
    python scripts/07_make_case_figure.py [PATIENT_ID]

If PATIENT_ID is omitted, it picks the first patient in series_index.csv
that has a usable CT+PET+RTSTRUCT match (same matching logic as the main
extraction script), so it should "just work" without arguments.

Output:
    outputs/figs/sample_slice_overlay.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from rt_utils import RTStructBuilder


def series_uid_of_dir(dicom_dir):
    import pydicom
    f = next(Path(dicom_dir).glob("*.dcm"))
    return pydicom.dcmread(str(f), stop_before_pixels=True).SeriesInstanceUID


def rtstruct_referenced_series_uid(rt_path):
    import pydicom
    ds = pydicom.dcmread(str(rt_path), stop_before_pixels=True)
    try:
        roi_seq = ds.ROIContourSequence[0]
        contour_seq = roi_seq.ContourSequence[0]
        ref_sop = contour_seq.ContourImageSequence[0].ReferencedSOPInstanceUID
        return ref_sop
    except Exception:
        return None


def read_series_as_image(dicom_dir):
    reader = sitk.ImageSeriesReader()
    file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    reader.SetFileNames(file_names)
    return reader.Execute()


def find_matching_ct_and_mask(pid, idx):
    p_rows = idx[idx["PatientID"] == pid]
    ct_rows = p_rows[p_rows["Modality"] == "CT"]
    rt_rows = p_rows[p_rows["Modality"] == "RTSTRUCT"]
    pt_rows = p_rows[p_rows["Modality"] == "PT"]
    if ct_rows.empty or rt_rows.empty or pt_rows.empty:
        return None

    for _, ct_row in ct_rows.iterrows():
        ct_dir = ct_row["dir"]
        try:
            ct_uid = series_uid_of_dir(ct_dir)
        except Exception:
            continue
        for _, rt_row in rt_rows.iterrows():
            rt_dir = Path(rt_row["dir"])
            dcm_files = list(rt_dir.glob("*.dcm"))
            if not dcm_files:
                continue
            try:
                rtstruct = RTStructBuilder.create_from(
                    dicom_series_path=str(ct_dir), rt_struct_path=str(dcm_files[0])
                )
            except Exception:
                continue
            roi_names = rtstruct.get_roi_names()
            match = next((n for n in roi_names if n in config.ROI_NAME_CANDIDATES), None)
            if match is None:
                match = roi_names[0] if roi_names else None
            if match is None:
                continue
            try:
                ct_img = read_series_as_image(ct_dir)
                mask_xyz = rtstruct.get_roi_mask_by_name(match)
                mask_zyx = np.transpose(mask_xyz, (2, 0, 1)).astype(np.uint8)
                if mask_zyx.shape != sitk.GetArrayFromImage(ct_img).shape:
                    continue
            except Exception:
                continue
            return {
                "ct_dir": ct_dir, "pt_dir": pt_rows.iloc[0]["dir"],
                "ct_img": ct_img, "mask_zyx": mask_zyx, "roi_name": match,
            }
    return None


def main():
    if not config.SERIES_INDEX_CSV.exists():
        print("ERROR: run 00_scan_dicom.py first.")
        sys.exit(1)

    idx = pd.read_csv(config.SERIES_INDEX_CSV)
    patients = sorted(idx["PatientID"].unique())

    requested = sys.argv[1] if len(sys.argv) > 1 else None
    candidates = [requested] if requested else patients

    found = None
    for pid in candidates:
        print(f"Trying {pid} ...")
        result = find_matching_ct_and_mask(pid, idx)
        if result is not None:
            found = (pid, result)
            break
    if found is None:
        print("Could not find a patient with a clean CT+PET+RTSTRUCT match.")
        sys.exit(1)

    pid, r = found
    print(f"Using patient {pid}, ROI '{r['roi_name']}'")

    ct_arr = sitk.GetArrayFromImage(r["ct_img"])          # (z, y, x)
    mask_arr = r["mask_zyx"]

    pt_img = read_series_as_image(r["pt_dir"])
    pt_on_ct = sitk.Resample(pt_img, r["ct_img"], sitk.Transform(),
                              sitk.sitkLinear, 0, pt_img.GetPixelID())
    pt_arr = sitk.GetArrayFromImage(pt_on_ct)

    # pick the axial slice with the largest tumor cross-section
    slice_areas = mask_arr.sum(axis=(1, 2))
    z = int(np.argmax(slice_areas))
    print(f"Largest-tumor slice: z={z} ({slice_areas[z]} voxels)")

    ct_slice = ct_arr[z]
    pt_slice = pt_arr[z]
    mask_slice = mask_arr[z]

    out_dir = config.OUTPUT_DIR / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    axes[0].imshow(ct_slice, cmap="gray", vmin=-150, vmax=250)
    axes[0].contour(mask_slice, colors="lime", linewidths=1.5)
    axes[0].set_title(f"CT + tumor contour\nPatient {pid}, slice {z}")
    axes[0].axis("off")

    axes[1].imshow(pt_slice, cmap="hot")
    axes[1].contour(mask_slice, colors="cyan", linewidths=1.5)
    axes[1].set_title("PET + tumor contour")
    axes[1].axis("off")

    axes[2].imshow(ct_slice, cmap="gray", vmin=-150, vmax=250)
    axes[2].imshow(np.ma.masked_where(pt_slice < np.percentile(pt_slice, 60), pt_slice),
                    cmap="hot", alpha=0.55)
    axes[2].contour(mask_slice, colors="lime", linewidths=1.5)
    axes[2].set_title("CT/PET fusion + contour")
    axes[2].axis("off")
    axes[2].legend(handles=[Patch(edgecolor="lime", facecolor="none", label="RTSTRUCT contour")],
                    loc="lower right", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    out_path = out_dir / "sample_slice_overlay.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()