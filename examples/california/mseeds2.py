# %%
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import fsspec
import numpy as np
import pandas as pd
import zarr
from args import parse_args
from tqdm import tqdm

from utils import (
    filter_and_sort_mseeds,
    get_neighbors_within_radius,
    load_stations,
    merge_mseeds_stations,
    scan_mseeds,
    sort_by_priorities,
)

args = parse_args()
year = f"{args.year:04d}"
jday = f"{args.jday:03d}"
knn_dist = args.knn_dist
print(f"{year = }, {jday = }, {knn_dist = }")

protocol = args.protocol
token_file = args.token_file
bucket = args.bucket


# %%
mseeds = scan_mseeds(target_year=year, target_jday=jday)
mseeds = pd.DataFrame(mseeds)

# %%
valid_instruments = ["HH", "BH", "EH", "SH", "DP", "EP", "HN"]
valid_components = ["3", "2", "1", "E", "N", "Z"]

mseeds = filter_and_sort_mseeds(mseeds, valid_instruments, valid_components)

# %%
stations = load_stations()

# %%
mseeds = merge_mseeds_stations(mseeds, stations)

# %%
mseeds = sort_by_priorities(mseeds)
print(f"Before grouping: {len(mseeds) = }")
print(mseeds.head())
mseeds = mseeds.groupby(["year", "jday", "network", "station"]).first().reset_index()
print(f"After grouping: {len(mseeds) = }")
mseeds = mseeds.sort_values(["year", "jday", "network", "station"])

# %%
distances, indices = get_neighbors_within_radius(mseeds, radius_km=knn_dist)

# %%
mseeds["station_id"] = (
    mseeds["network"] + "." + mseeds["station"] + "." + mseeds["location"] + "." + mseeds["instrument"]
)
pairs_idx = [(i, j) for i in range(len(indices)) for j in indices[i] if i <= j]
pairs_sid = [(mseeds.iloc[i].station_id, mseeds.iloc[j].station_id) for i, j in pairs_idx]
sid2idx = {sid: idx for sid, idx in zip(pairs_sid, pairs_idx)}


# %%
ccf = None
try:
    store = zarr.storage.FsspecStore.from_url(
        f"gs://cctorch/ambient_noise/ccf/{year}/{year}.{jday}.zarr", read_only=True, storage_options={"anon": True}
    )
    ccf = zarr.open_group(store=store, mode="r")
except Exception as e:
    print(f"Error opening ccf: {e}")


def scan_ccf(s1):
    return [(s1, s2) for s2 in ccf[s1].keys()]


pairs_ccf = []
if ccf is not None:
    with ThreadPoolExecutor(max_workers=mp.cpu_count() * 2) as executor:
        ccf_keys = list(ccf.keys())
        futures = [executor.submit(scan_ccf, s1) for s1 in ccf_keys]
        for future in tqdm(futures, total=len(ccf_keys), desc="Scanning ccf"):
            pairs_ccf.extend(future.result())

print(f"Total pairs: {len(pairs_sid)}")
print(f"Processed pairs: {len(pairs_ccf)}")

# pairs_idx = [sid2idx[pair] for pair in pairs_sid if pair not in pairs_ccf]
pairs_sid = set(pairs_sid) - set(pairs_ccf)
pairs_sid = sorted(list(pairs_sid))
pairs_idx = [sid2idx[sid] for sid in pairs_sid]
print(f"Remaining pairs: {len(pairs_idx)}")

# %%
mseeds["file_name"].to_csv(f"mseeds2_{year}_{jday}.txt", index=False, header=True)

# %%
# with open(f"pairs2_{year}_{jday}.txt", "w") as f:
#     f.writelines(f"{i},{j}\n" for i in range(len(indices)) for j in indices[i] if i <= j)
with open(f"pairs2_{year}_{jday}.txt", "w") as f:
    f.writelines(f"{i},{j}\n" for i, j in pairs_idx)

# %%
fs = fsspec.filesystem(protocol, token=token_file)
fs.put(f"mseeds2_{year}_{jday}.txt", f"{bucket}/mseed_list/mseeds2_{year}_{jday}.txt")
fs.put(f"pairs2_{year}_{jday}.txt", f"{bucket}/mseed_list/pairs2_{year}_{jday}.txt")
print(f"mseeds2_{year}_{jday}.txt -> {bucket}/mseed_list/mseeds2_{year}_{jday}.txt")
print(f"pairs2_{year}_{jday}.txt -> {bucket}/mseed_list/pairs2_{year}_{jday}.txt")
