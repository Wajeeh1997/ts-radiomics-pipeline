# from idc_index import index
# client = index.IDCClient()
# print([m for m in dir(client) if not m.startswith("_")])
from idc_index import index
client = index.IDCClient()
df = client.index
sub = df[df["PatientID"] == "STS_041"]
print(sub[["SeriesInstanceUID", "Modality", "SeriesDescription", "collection_id"]].to_string())