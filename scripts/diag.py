import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pydicom
import pandas as pd
import config

idx = pd.read_csv(config.SERIES_INDEX_CSV)
pid = "STS_041"
p_rows = idx[idx["PatientID"] == pid]

ct_dir = p_rows[p_rows["Modality"] == "CT"].iloc[0]["dir"]
rt_dir = p_rows[p_rows["Modality"] == "RTSTRUCT"].iloc[0]["dir"]
print("CT dir:", ct_dir)
print("RTSTRUCT dir:", rt_dir)

rt_file = list(Path(rt_dir).glob("*.dcm"))[0]
ds = pydicom.dcmread(rt_file, stop_before_pixels=True)
ref_sops = {c.ContourImageSequence[0].ReferencedSOPInstanceUID
            for roi in ds.ROIContourSequence for c in roi.ContourSequence}
print("RTSTRUCT references", len(ref_sops), "unique CT slices")

on_disk = set()
for f in Path(ct_dir).glob("*.dcm"):
    d = pydicom.dcmread(f, stop_before_pixels=True)
    on_disk.add(d.SOPInstanceUID)
print("CT folder has", len(on_disk), "slices")
print("Missing:", len(ref_sops - on_disk))

how_many_ct_series = p_rows[p_rows["Modality"] == "CT"].shape[0]
print("CT series for this patient:", how_many_ct_series)

# --- extra diagnostics ---
one_ct_file = list(Path(ct_dir).glob("*.dcm"))[0]
ct_ds = pydicom.dcmread(one_ct_file, stop_before_pixels=True)
print("CT SeriesInstanceUID:", ct_ds.SeriesInstanceUID)
print("CT StudyInstanceUID:", ct_ds.StudyInstanceUID)

try:
    ref_series_uid = (ds.ReferencedFrameOfReferenceSequence[0]
                         .RTReferencedStudySequence[0]
                         .RTReferencedSeriesSequence[0]
                         .SeriesInstanceUID)
    print("RTSTRUCT's referenced SeriesInstanceUID:", ref_series_uid)
except Exception as e:
    print("Could not read referenced series UID:", e)

one_missing_sop = list(ref_sops - on_disk)[0]
print("Example missing SOP UID:", one_missing_sop)