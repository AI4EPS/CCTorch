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
def parse_fname(path_fname, region="NC"):
    try:
        if region == "NC":
            station, network, channel, location, D, year, jday = path_fname.split("/")[-1].split(".")
            instrument = channel[:2]
            component = channel[2]

        if region == "SC":
            fname = path_fname.split("/")[-1]  #
            network = fname[:2]
            station = fname[2:7].rstrip("_")
            instrument = fname[7:9]
            component = fname[9]
            channel = f"{instrument}{component}"
            location = fname[10:12].rstrip("_")
            year = fname[13:17]
            jday = fname[17:20]

        return {
            "file_name": f"s3://{path_fname}",
            "station": station,
            "network": network,
            "location": location,
            "instrument": instrument,
            "component": component,
            "channel": channel,
            "year": year,
            "jday": jday,
        }
    except:
        return None


# %%
def scan_mseeds_nc(target_year="2024", target_jday="001"):
    """
    Get all file names of continuous waveforms in the given S3 path.
    """

    fs = fsspec.filesystem("s3", anon=True)

    networks = []
    for root, dirs, files in fs.walk(f"s3://ncedc-pds/continuous_waveforms/", maxdepth=1):
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

    mseeds = [parse_fname(mseed, region="NC") for mseed in mseeds]
    return mseeds


def scan_mseeds_sc(target_year="2024", target_jday="001"):
    """
    Get all file names of continuous waveforms in the given S3 path.
    """

    fs = fsspec.filesystem("s3", anon=True)

    years = []
    for root, dirs, files in fs.walk(f"s3://scedc-pds/continuous_waveforms", maxdepth=1):
        years.extend([f"{root}/{d}" for d in dirs])

    years = [y for y in years if y.endswith(target_year)]  ## TODO: Include all years

    days = []
    mseeds = []
    for year in years:
        for root, dirs, files in fs.walk(f"{year}", maxdepth=1):
            if dirs:
                days.extend([f"{root}/{d}" for d in dirs])
            else:
                mseeds.extend([f"{root}/{f}" for f in files])

    days = [d for d in days if d.endswith(target_jday)]  ## TODO: Include all days
    if days:
        for day in days:
            for root, dirs, files in fs.walk(f"{day}", maxdepth=1):
                mseeds.extend([f"{root}/{f}" for f in files])

    mseeds = [parse_fname(mseed, region="SC") for mseed in mseeds]
    return mseeds


def get_neighbors_within_radius(df, radius_km=100):

    proj = Proj(proj="aeqd", lat_0=df["latitude"].mean(), lon_0=df["longitude"].mean())
    coords = proj(df["longitude"], df["latitude"], inverse=True)
    coords = np.array(coords).T

    knn = NearestNeighbors(radius=radius_km, metric="haversine")
    knn.fit(coords)

    distances, indices = knn.radius_neighbors(coords, radius=radius_km, return_distance=True)

    return distances, indices


mseeds_nc = scan_mseeds_nc(target_year=year, target_jday=jday)
mseeds_sc = scan_mseeds_sc(target_year=year, target_jday=jday)
print(f"NC: {len(mseeds_nc)}")
print(f"SC: {len(mseeds_sc)}")
mseeds = pd.DataFrame(mseeds_nc + mseeds_sc)

# %%
region = "NC"  # %%
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
stations = stations.fillna({"location": ""})
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
mseeds["file_name"].to_csv(f"mseeds1_{year}_{jday}.txt", index=False, header=True)

# %%
with open(f"pairs1_{year}_{jday}.txt", "w") as f:
    f.writelines(f"{i},{j}\n" for i in range(len(indices)) for j in indices[i])

# %%
fs = fsspec.filesystem(protocol, token=token_file)
fs.put(f"mseeds1_{year}_{jday}.txt", f"{bucket}/mseed_list/mseeds1_{year}_{jday}.txt")
fs.put(f"pairs1_{year}_{jday}.txt", f"{bucket}/mseed_list/pairs1_{year}_{jday}.txt")
