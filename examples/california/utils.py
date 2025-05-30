import fsspec
import numpy as np
import pandas as pd
from pyproj import Proj
from sklearn.neighbors import NearestNeighbors


def parse_fname(path_fname, region=None):
    try:
        if region == "NC":
            station, network, channel, location, D, year, jday = path_fname.split("/")[-1].split(".")
            instrument = channel[:2]
            component = channel[2]
            file_name = f"s3://{path_fname}"

        elif region == "SC":
            fname = path_fname.split("/")[-1]  #
            network = fname[:2]
            station = fname[2:7].rstrip("_")
            instrument = fname[7:9]
            component = fname[9]
            channel = f"{instrument}{component}"
            location = fname[10:12].rstrip("_")
            year = fname[13:17]
            jday = fname[17:20]
            file_name = f"s3://{path_fname}"

        else:
            tmp = path_fname.split("/")
            network, station, location, channel, _ = tmp[-1].split(".")
            instrument = channel[:2]
            component = channel[2]
            year = tmp[-3]
            jday = tmp[-2]
            file_name = f"gs://{path_fname}"

        return {
            "file_name": file_name,
            "station": station,
            "network": network,
            "location": location,
            "instrument": instrument,
            "component": component,
            "channel": channel,
            "year": year,
            "jday": jday,
        }
    except Exception as e:
        print(f"Error parsing {path_fname}: {e}")
        return None


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
    """Get neighbors within a given radius for each point in the dataframe."""
    proj = Proj(proj="aeqd", lat_0=df["latitude"].mean(), lon_0=df["longitude"].mean(), units="km")
    coords = proj(df["longitude"], df["latitude"])
    coords = np.array(coords).T

    knn = NearestNeighbors(radius=radius_km)
    knn.fit(coords)

    distances, indices = knn.radius_neighbors(coords, radius=radius_km, return_distance=True)

    return distances, indices


def filter_and_sort_mseeds(mseeds_df, valid_instruments, valid_components):
    """Filter and sort mseeds dataframe based on valid instruments and components."""
    mseeds = mseeds_df.copy()

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

    return mseeds


def load_stations():
    """Load and process stations data."""
    stations = pd.read_csv(
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

    return stations


def merge_mseeds_stations(mseeds, stations):
    """Merge mseeds and stations dataframes."""
    mseeds = mseeds.fillna({"location": ""})
    stations = stations.fillna({"location": ""})
    mseeds = mseeds.astype({col: str for col in ["network", "station", "instrument", "location"]})
    stations = stations.astype({col: str for col in ["network", "station", "instrument", "location"]})

    mseeds = mseeds.merge(
        stations[["network", "station", "location", "instrument", "longitude", "latitude"]],
        on=["network", "station", "location", "instrument"],
        how="inner",
    )

    return mseeds


def sort_by_priorities(mseeds):
    """Sort mseeds by location and instrument priorities."""
    location_priorities = (
        "",
        "00",
        "10",
        "01",
        "20",
        "02",
        "30",
        "03",
        "40",
        "04",
        "50",
        "05",
        "60",
        "06",
        "70",
        "07",
        "80",
        "08",
        "90",
        "09",
    )
    location_priority_map = {loc: i for i, loc in enumerate(location_priorities)}

    instrument_priorities = ("HH", "BH", "MH", "EH", "LH", "HL", "BL", "ML", "EL", "LL", "SH")
    instrument_priority_map = {ch: i for i, ch in enumerate(instrument_priorities)}

    mseeds["location_priority"] = mseeds["location"].map(location_priority_map)
    mseeds["instrument_priority"] = mseeds["instrument"].apply(
        lambda x: instrument_priority_map.get(x, len(instrument_priorities))
    )

    mseeds.sort_values(["year", "jday", "network", "station", "location_priority", "instrument_priority"], inplace=True)
    mseeds.drop(["location_priority", "instrument_priority"], axis=1, inplace=True)

    return mseeds


def save_and_upload_files(mseeds, pairs_idx, year, jday, protocol, token_file, bucket):
    """Save files locally and upload to storage."""
    # Save mseeds file
    mseeds["file_name"].to_csv(f"mseeds_{year}_{jday}.txt", index=False, header=True)

    # Save pairs file
    with open(f"pairs_{year}_{jday}.txt", "w") as f:
        f.writelines(f"{i},{j}\n" for i, j in pairs_idx)

    # Upload files
    fs = fsspec.filesystem(protocol, token=token_file)
    fs.put(f"mseeds_{year}_{jday}.txt", f"{bucket}/mseed_list/mseeds_{year}_{jday}.txt")
    fs.put(f"pairs_{year}_{jday}.txt", f"{bucket}/mseed_list/pairs_{year}_{jday}.txt")

    print(f"mseeds_{year}_{jday}.txt -> {bucket}/mseed_list/mseeds_{year}_{jday}.txt")
    print(f"pairs_{year}_{jday}.txt -> {bucket}/mseed_list/pairs_{year}_{jday}.txt")
