import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
from rt_utils import RTStructBuilder
import config

pid = "STS_003"
idx = pd.read_csv(config.SERIES_INDEX_CSV)
p_rows = idx[idx["PatientID"] == pid]

ct_rows = p_rows[p_rows["Modality"] == "CT"]
pt_rows = p_rows[p_rows["Modality"] == "PT"]
rt_rows = p_rows[p_rows["Modality"] == "RTSTRUCT"]


def series_uid_of_dir(dicom_dir):
    dcm_files = list(Path(dicom_dir).glob("*.dcm"))
    if not dcm_files:
        return None
    ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
    return getattr(ds, "SeriesInstanceUID", None)


def rtstruct_referenced_series_uid(rt_path):
    ds = pydicom.dcmread(rt_path, stop_before_pixels=True)
    try:
        return (ds.ReferencedFrameOfReferenceSequence[0]
                  .RTReferencedStudySequence[0]
                  .RTReferencedSeriesSequence[0]
                  .SeriesInstanceUID)
    except Exception:
        return None


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
        if ref_uid == candidate_ct_uid:
            rt_dir = candidate_rt_dir
            ct_dir = candidate_ct_dir
            break
    if rt_dir is not None:
        break

print("CT dir:", ct_dir)
print("RTSTRUCT dir:", rt_dir)

reader = sitk.ImageSeriesReader()
reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(ct_dir)))
ct_img = reader.Execute()

rt_file = list(Path(rt_dir).glob("*.dcm"))[0]
rtstruct = RTStructBuilder.create_from(dicom_series_path=str(ct_dir), rt_struct_path=str(rt_file))
roi_names = rtstruct.get_roi_names()
match = next((n for n in roi_names if n in config.ROI_NAME_CANDIDATES), roi_names[0])
mask_xyz = rtstruct.get_roi_mask_by_name(match)
mask_zyx = np.transpose(mask_xyz, (2, 0, 1)).astype(np.uint8)

mask_img = sitk.GetImageFromArray(mask_zyx)
mask_img.CopyInformation(ct_img)

pt_dir = pt_rows.iloc[0]["dir"]
reader2 = sitk.ImageSeriesReader()
reader2.SetFileNames(reader2.GetGDCMSeriesFileNames(str(pt_dir)))
pt_img = reader2.Execute()

pt_mask_img = sitk.Resample(mask_img, pt_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask_img.GetPixelID())

pt_arr = sitk.GetArrayFromImage(pt_img)
mask_arr = sitk.GetArrayFromImage(pt_mask_img)

vals = pt_arr[mask_arr == 1]
print("PET dtype:", pt_arr.dtype)
print("PET voxel count in mask:", vals.size)
print("PET value min/max inside mask:", vals.min(), vals.max())
print("PET value range inside mask:", vals.max() - vals.min())
print("Estimated GLCM gray levels at binWidth=0.25:", (vals.max() - vals.min()) / 0.25)
print("PET overall image min/max:", pt_arr.min(), pt_arr.max())