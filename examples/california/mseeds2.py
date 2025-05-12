# %%
import fsspec
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import BallTree, NearestNeighbors
import numpy as np
from pyproj import Proj

from args import parse_args


args = parse_args()
year = f"{args.year:04d}"
jday = f"{args.jday:03d}"
knn_dist = args.knn_dist

protocol = args.protocol
token_file = args.token_file
bucket = args.bucket


# %%
def parse_fname(path_fname):

    tmp = path_fname.split("/")
    network, station, location, channel, _ = tmp[-1].split(".")
    instrument = channel[:2]
    component = channel[2]
    year = tmp[-3]
    jday = tmp[-2]

    return {
        "file_name": f"gs://{path_fname}",
        "station": station,
        "network": network,
        "location": location,
        "instrument": instrument,
        "component": component,
        "channel": channel,
        "year": year,
        "jday": jday,
    }


# %%
def scan_mseeds(target_year="2024", target_jday="001"):
    """
    Get all file names of continuous waveforms in the given S3 path.
    """

    fs = fsspec.filesystem("gs", anon=True)

    networks = []
    for root, dirs, files in fs.walk(f"gs://cctorch/ambient_noise/waveforms", maxdepth=1):
        networks.extend([f"{root}/{d}" for d in dirs])

    years = []
    for network in networks:
        for root, dirs, files in fs.walk(f"{network}", maxdepth=1):
            years.extend([f"{root}/{d}" for d in dirs])

    years = [y for y in years if y.endswith(target_year)]  ## TODO: Include all years

    days = []
    for year in years:
        for root, dirs, files in fs.walk(f"{year}", maxdepth=1):
            days.extend([f"{root}/{d}" for d in dirs])

    days = [d for d in days if d.endswith(target_jday)]  ## TODO: Include all days

    mseeds = []
    for day in days:
        for root, dirs, files in fs.walk(f"{day}", maxdepth=1):
            mseeds.extend([f"{root}/{f}" for f in files])

    mseeds = [parse_fname(mseed) for mseed in mseeds]
    return mseeds


def get_neighbors_within_radius(df, radius_km=100):

    proj = Proj(proj="aeqd", lat_0=df["latitude"].mean(), lon_0=df["longitude"].mean())
    coords = proj(df["longitude"], df["latitude"], inverse=True)
    coords = np.array(coords).T

    knn = NearestNeighbors(radius=radius_km, metric="haversine")
    knn.fit(coords)

    distances, indices = knn.radius_neighbors(coords, radius=radius_km, return_distance=True)

    return distances, indices


mseeds = scan_mseeds(target_year=year, target_jday=jday)
mseeds = pd.DataFrame(mseeds)

# %%
valid_instruments = ["HH", "BH", "EH"]  # ["SH", "DP", "EP", "HN"]
valid_components = ["3", "2", "1", "E", "N", "Z"]

mseeds = mseeds[mseeds["instrument"].isin(valid_instruments)]
mseeds = mseeds[mseeds["component"].isin(valid_components)]

mseeds["instrument"] = pd.Categorical(mseeds["instrument"], categories=valid_instruments, ordered=True)
mseeds = mseeds.sort_values(by=["station", "instrument"]).reset_index(drop=True)
mseeds = mseeds.drop_duplicates(
    subset=["station", "network", "location", "instrument", "component", "channel", "year", "jday"]
)


mseeds = mseeds.groupby(["year", "jday", "network", "station", "location", "instrument"], observed=True).agg(
    file_name=("file_name", lambda x: "|".join(sorted(x)))
)
mseeds = mseeds.reset_index()

mseeds = mseeds.drop_duplicates(subset="station", keep="first")
# %%
stations = pd.read_csv(
    # "stations.csv",
    f"gs://cctorch/ambient_noise/network/stations.csv",
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
distances, indices = get_neighbors_within_radius(mseeds, radius_km=knn_dist)

# %%
mseeds["file_name"].to_csv(f"mseeds2_{year}_{jday}.txt", index=False, header=True)

# %%
with open(f"pairs2_{year}_{jday}.txt", "w") as f:
    f.writelines(f"{i},{j}\n" for i in range(len(indices)) for j in indices[i])

# %%
fs = fsspec.filesystem(protocol, token=token_file)
fs.put(f"mseeds2_{year}_{jday}.txt", f"{bucket}/mseed_list/mseeds2_{year}_{jday}.txt")
fs.put(f"pairs2_{year}_{jday}.txt", f"{bucket}/mseed_list/pairs2_{year}_{jday}.txt")
