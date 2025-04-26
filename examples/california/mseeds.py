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
        instrument = channel[:2]
        component = channel[2]

    if region == "SC":
        fname = fname.split("/")[-1]  #
        network = fname[:2]
        station = fname[2:7].rstrip("_")
        instrument = fname[7:9]
        component = fname[9]
        channel = f"{instrument}{component}"
        location = fname[10:12].rstrip("_")
        year = fname[13:17]
        jday = fname[17:20]

    return {
        "file_name": f"s3://{fname}",
        "station": station,
        "network": network,
        "location": location,
        "instrument": instrument,
        "component": component,
        "channel": channel,
        "year": year,
        "jday": jday,
    }


mseeds = [parse_fname(mseed) for mseed in mseeds]
mseeds = pd.DataFrame(mseeds)
# print(mseeds)


# %%
region = "NC"  # %%
valid_instruments = ["BH", "HH", "EH", "HN", "DP", "SH", "EP"]
valid_components = ["3", "2", "1", "E", "N", "Z"]
mseeds = mseeds[mseeds["instrument"].isin(valid_instruments)]
mseeds = mseeds[mseeds["component"].isin(valid_components)]
mseeds = mseeds.groupby(["year", "jday", "network", "station", "location", "instrument"]).agg(
    file_name=("file_name", lambda x: "|".join(sorted(x)))
)
mseeds = mseeds.reset_index()


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

stations["instrument"] = stations["channel"].str[:2]
stations["component"] = stations["channel"].str[2]
stations = stations.groupby(["network", "station", "location", "instrument"]).agg(
    {"longitude": "first", "latitude": "first", "elevation_m": "first", "sensitivity": "first"}
)
stations = stations.reset_index()

# %%
mseeds = mseeds.fillna({"location": ""})
stations = stations.fillna({"location": ""})
mseeds = mseeds.astype({col: str for col in ["network", "station", "instrument", "location"]})
stations = stations.astype({col: str for col in ["network", "station", "instrument", "location"]})

# print(mseeds)
# print(stations)
# %%
mseeds = mseeds.merge(
    stations[["network", "station", "location", "instrument", "longitude", "latitude"]],
    on=[
        "network",
        "station",
        "location",
        "instrument",
    ],
    how="inner",
)

# %%
n_neighbors = 100  ## TODO: optimize the pairs
knn = NearestNeighbors(n_neighbors=n_neighbors)
knn.fit(mseeds[["longitude", "latitude"]])

# %%
distances, indices = knn.kneighbors(mseeds[["longitude", "latitude"]])

# %%
mseeds["file_name"].to_csv("mseeds.txt", index=False, header=True)

# %%
with open("pairs.txt", "w") as f:
    f.writelines(f"{i},{j}\n" for i in range(len(indices)) for j in indices[i])
