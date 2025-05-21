# %%
import fsspec
import pandas as pd
from args import parse_args
from tqdm import tqdm

from utils import (
    filter_and_sort_mseeds,
    get_neighbors_within_radius,
    load_stations,
    merge_mseeds_stations,
    scan_mseeds,
    scan_mseeds_nc,
    scan_mseeds_sc,
    sort_by_priorities,
)

args = parse_args()
year = f"{args.year:04d}"
jday = f"{args.jday:03d}"
knn_dist = args.knn_dist

protocol = args.protocol
token_file = args.token_file
bucket = args.bucket

# %%

mseeds_nc = scan_mseeds_nc(target_year=year, target_jday=jday)
mseeds_sc = scan_mseeds_sc(target_year=year, target_jday=jday)
print(f"NC: {len(mseeds_nc)}")
print(f"SC: {len(mseeds_sc)}")
mseeds = pd.DataFrame(mseeds_nc + mseeds_sc)


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
processed = scan_mseeds(target_year=year, target_jday=jday)
processed = pd.DataFrame(processed)
if len(processed) > 0:
    print(f"Before filtering: {len(processed)}")
    print(mseeds.head())
    print(processed.head())
    mseeds = mseeds[
        ~mseeds.set_index(["year", "jday", "network", "station"]).index.isin(
            processed.set_index(["year", "jday", "network", "station"]).index
        )
    ]
    print(f"After filtering: {len(mseeds) = }")
    print(mseeds.head())

# %%
if len(mseeds) > 0:
    distances, indices = get_neighbors_within_radius(mseeds, radius_km=knn_dist)
else:
    distances = []
    indices = []

# %%
mseeds["file_name"].to_csv(f"mseeds1_{year}_{jday}.txt", index=False, header=True)

# %%
with open(f"pairs1_{year}_{jday}.txt", "w") as f:
    f.writelines(f"{i},{j}\n" for i in range(len(indices)) for j in indices[i] if i <= j)

# %%
fs = fsspec.filesystem(protocol, token=token_file)
fs.put(f"mseeds1_{year}_{jday}.txt", f"{bucket}/mseed_list/mseeds1_{year}_{jday}.txt")
fs.put(f"pairs1_{year}_{jday}.txt", f"{bucket}/mseed_list/pairs1_{year}_{jday}.txt")
print(f"mseeds1_{year}_{jday}.txt -> {bucket}/mseed_list/mseeds1_{year}_{jday}.txt")
print(f"pairs1_{year}_{jday}.txt -> {bucket}/mseed_list/pairs1_{year}_{jday}.txt")
