# %%
import fsspec
import json
import os
import h5py

# %%
token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
with open(token_json, "r") as fp:
    token = json.load(fp)
fs = fsspec.filesystem("gs", token=token)

# %%
ccf_file = "gs://cctorch/ambient_noise/ccf/2024/2024.001.h5"

with fs.open(ccf_file, "rb") as fp:
    ccf = h5py.File(fp, "r")
    print(ccf.keys())
    raise

#


# %%
