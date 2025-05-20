# %%
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import fsspec
import h5py
import numpy as np
import tqdm
import zarr


def create_array(store, name, data):
    data = zarr.create_array(store, name=name.replace(".npy", ""), data=data, overwrite=True)
    return


# %%
if __name__ == "__main__":

    # %%
    store = zarr.storage.FsspecStore.from_url(
        "gs://cctorch/ambient_noise/ccf/2024/2024.001.zarr", read_only=True, storage_options={"anon": True}
    )
    ccf = zarr.open_group(store=store, mode="r")

    print(f"{len(list(ccf.keys())) = }")
    for s1 in ccf.keys():
        print(f"{len(list(ccf[s1].keys())) = }")
        for s2 in ccf[s1].keys():
            print(s1, s2)
            print(ccf[s1][s2].shape)
            print(dict(ccf[s1][s2].attrs))
            break
        break

    # %%
    ccf_file = "gs://cctorch/ambient_noise/ccf/2024/2024.001.h5"
    # fs = fsspec.filesystem("gs", token=token)
    # with fs.open(ccf_file, "rb") as fp:
    with fsspec.open(ccf_file, "rb", anon=True) as fp:
        ccf = h5py.File(fp, "r")
        for s1 in ccf.keys():
            for s2 in ccf[s1].keys():
                print(s1, s2)
                print(ccf[s1][s2]["xcorr"].shape)
                print(dict(ccf[s1][s2]["xcorr"].attrs))
                break
            break

    # %%
