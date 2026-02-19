# %%
import fsspec
import numpy as np
import pandas as pd
import zarr
from args import parse_args

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

local_station_file = args.local_station_file


# %%
mseeds = scan_mseeds(target_year=year, target_jday=jday)
mseeds = pd.DataFrame(mseeds)

# %%
valid_instruments = ["HH", "BH", "EH", "SH", "DP", "EP", "HN"]
valid_components = ["3", "2", "1", "E", "N", "Z"]

if local_station_file:
    print("Filtering mseeds using local station list")
    nw_st_df = pd.read_csv(local_station_file)
    target_nw = nw_st_df['network'].drop_duplicates().tolist()
    target_st = nw_st_df['station'].drop_duplicates().tolist()
    valid_instruments = nw_st_df['instrument'].drop_duplicates().tolist()

    mseeds_filt = mseeds.copy()
    mseeds_filt = mseeds_filt[mseeds_filt["network"].isin(target_nw)]
    mseeds = mseeds_filt[mseeds_filt["station"].isin(target_st)]

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
print(f"Getting neighbors within {knn_dist} km")
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
store = None
try:
    store = zarr.storage.FsspecStore.from_url(
        f"gs://cctorch/ambient_noise/ccf/{year}/{year}.{jday}.zarr", read_only=True, storage_options={"anon": True}
    )
    ccf = zarr.open_group(store=store, mode="r")
except Exception as e:
    print(f"Error opening ccf: {e}")


pairs_ccf = []
if ccf is not None and "id1" in ccf and "id2" in ccf:
    pairs_ccf = list(zip(ccf["id1"][:], ccf["id2"][:]))
if store is not None:
    store.close()

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
fs = fsspec.filesystem(protocol, token='google_default') # hotfix for credential issue with token_file
fs.put(f"mseeds2_{year}_{jday}.txt", f"{bucket}/mseed_list/mseeds2_{year}_{jday}.txt")
fs.put(f"pairs2_{year}_{jday}.txt", f"{bucket}/mseed_list/pairs2_{year}_{jday}.txt")
print(f"mseeds2_{year}_{jday}.txt -> {bucket}/mseed_list/mseeds2_{year}_{jday}.txt")
print(f"pairs2_{year}_{jday}.txt -> {bucket}/mseed_list/pairs2_{year}_{jday}.txt")
