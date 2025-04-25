# %%
import concurrent.futures
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
import obspy
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


# %%
fs = fsspec.filesystem("s3", anon=True)

# %%
folders = []
networks = []
for root, dirs, files in fs.walk(f"s3://ncedc-pds/continuous_waveforms/", maxdepth=1):
    networks.extend([f"{root}/{d}" for d in dirs])

years = []
for network in networks:
    for root, dirs, files in fs.walk(f"{network}", maxdepth=1):
        years.extend([f"{root}/{d}" for d in dirs])

years = [y for y in years if y.endswith("2024")]  ## TODO: Include all years

days = []
for year in years:
    for root, dirs, files in fs.walk(f"{year}", maxdepth=1):
        days.extend([f"{root}/{d}" for d in dirs])

days = [d for d in days if d.endswith("001")]  ## TODO: Include all days

mseeds = []
for day in days:
    for root, dirs, files in fs.walk(f"{day}", maxdepth=1):
        mseeds.extend([f"{root}/{f}" for f in files])


# %%
def parse_fname(fname, region="NC"):
    if region == "NC":
        station, network, channel, location, D, year, jday = fname.split("/")[-1].split(".")

    return {
        "file_name": f"s3://{fname}",
        "station": station,
        "network": network,
        "channel": channel,
        "location": location,
        "year": year,
        "jday": jday,
    }


mseeds = [parse_fname(mseed) for mseed in mseeds]
mseeds = pd.DataFrame(mseeds)

# %%
stations = pd.read_csv(
    "stations.csv",
    dtype={
        "network": str,
        "station": str,
        "location": str,
        "channel": str,
        "longitude": float,
        "latitude": float,
        "elevation_m": float,
        "sensitivity": float,
    },
)

# %%
mseeds = mseeds.fillna({"location": ""})
stations = stations.fillna({"location": ""})
mseeds = mseeds.astype({col: str for col in ["network", "station", "channel", "location"]})
stations = stations.astype({col: str for col in ["network", "station", "channel", "location"]})

# %%
mseeds_ = mseeds.merge(
    stations[["network", "station", "location", "channel", "longitude", "latitude"]],
    on=["network", "station", "location", "channel"],
    how="inner",
)

# %%
n_neighbors = 100  ## TODO: optimize the pairs
knn = NearestNeighbors(n_neighbors=n_neighbors)
knn.fit(mseeds_[["longitude", "latitude"]])

# %%
distances, indices = knn.kneighbors(mseeds_[["longitude", "latitude"]])

# %%
mseeds_["file_name"].to_csv("mseeds.txt", index=False, header=True)

# %%
with open("pairs.txt", "w") as f:
    f.writelines(f"{i},{j}\n" for i in range(len(indices)) for j in indices[i])
