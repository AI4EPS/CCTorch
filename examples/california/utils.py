import fsspec
import numpy as np
import pandas as pd
from pyproj import Proj
from sklearn.neighbors import NearestNeighbors

import logging
from obspy import read_inventory
from obspy.core import UTCDateTime
import scipy.fft as sf
from scipy.fft import next_fast_len

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


def get_instrument_resp(network, station):
    fs_s3 = fsspec.filesystem('s3')
    file_instru_resp = f"s3://ncedc-pds/FDSNstationXML/{network}/{network}.{station}.xml"
    try:
        with fs_s3.open(file_instru_resp, "rb") as f:
            return read_inventory(f)
    except:
        if network in ["CI", "ZY"]:
            network_folder = "CI"
        else:
            network_folder = "unauthoritative-XML"
        file_instru_resp = f"s3://scedc-pds/FDSNstationXML/{network_folder}/{network}_{station}.xml"
        with fs_s3.open(file_instru_resp, "rb") as f:
            return read_inventory(f)

def get_nc_info(fname):
    file = fname.split('/')[-1]
    network = file.split(".")[1]
    station = file.split(".")[0]
    channel = file.split(".")[2]
    location = file.split(".")[3]
    year = file.split(".")[-2]
    jday = file.split(".")[-1]
    date_target = obspy.UTCDateTime(year=int(year), julday=int(jday))
    channel_id = f"{network}.{station}.{location}.{channel}"
    return network, station, channel_id, date_target

def get_sc_info(fname):
    file = fname.split('/')[-1]
    network = file[:2]
    station = file[2:7].strip("_")
    channel = file[7:10]
    year, jday = int(file[-10:-6]), int(file[-6:-3])
    date_target = obspy.UTCDateTime(year=year, julday=jday)
    channel_id = f"{network}.{station}..{channel}"
    return network, station, channel_id, date_target

def get_instrument_response(fname):
    print(f"Finding instrument response for {fname}...")
    if fname.startswith("s3://ncedc-pds"):
        network, station, channel_id, date_target = get_nc_info(fname)   
    elif fname.startswith("s3://scedc-pds"):
        network, station, channel_id, date_target = get_sc_info(fname)
    else:
        raise ValueError(f"Unsupported file format: {fname}")
    return get_instrument_resp(network, station), channel_id, date_target

def from_inv_to_pd(inv, channel_id, date_target):
    channels = []
    for net in inv.networks:
        for sta in net.stations:
            for cha in sta.channels:
                try:
                    seed_id = "%s.%s.%s.%s" % (net.code, sta.code,
                                                cha.location_code,
                                                cha.code)
                    resp = inv.get_response(seed_id, cha.start_date+10)
                    polezerostage = resp.get_paz()
                    totalsensitivity = resp.instrument_sensitivity
                    pzdict = {}
                    pzdict['poles'] = polezerostage.poles
                    pzdict['zeros'] = polezerostage.zeros
                    pzdict['gain'] = polezerostage.normalization_factor
                    pzdict['sensitivity'] = totalsensitivity.value
                    channels.append([seed_id, cha.start_date,
                                        cha.end_date or UTCDateTime(),
                                        pzdict, cha.latitude,
                                        cha.longitude])
                except:
                    continue
    channels = pd.DataFrame(channels, columns=["channel_id", "start_date",
                                               "end_date", "paz", "latitude",
                                               "longitude"],)
    channels = channels[channels["channel_id"] == channel_id]
    if len(channels) > 1:
        channels = channels[channels["start_date"] <= date_target]
    if len(channels) > 1:
        channels = channels[channels["end_date"] >= date_target]
    elif len(channels) == 0:
        raise ValueError(f"No channel found for {channel_id} at {date_target}")
    return channels

def get_response_paz(fname):
    inv, channel_id, date_target = get_instrument_response(fname)
    response = from_inv_to_pd(inv, channel_id, date_target)
    return response["paz"].values[0]

def check_and_phase_shift(trace, taper_length=20.0):
    # TODO replace this hard coded taper length
    if trace.stats.npts < 4 * taper_length*trace.stats.sampling_rate:
        trace.data = np.zeros(trace.stats.npts)
        return trace

    dt = np.mod(trace.stats.starttime.datetime.microsecond*1.0e-6,
                trace.stats.delta)
    if (trace.stats.delta - dt) <= np.finfo(float).eps:
        dt = 0.
    if dt != 0.:
        if dt <= (trace.stats.delta / 2.):
            dt = -dt
        else:
            dt = (trace.stats.delta - dt)
        logging.debug("correcting time by %.6fs"%dt)
        trace.detrend(type="demean")
        trace.detrend(type="simple")
        trace.taper(max_percentage=None, max_length=1.0)
        n = next_fast_len(int(trace.stats.npts))
        FFTdata = sf.fft(trace.data, n=n)
        fftfreq = sf.fftfreq(n, d=trace.stats.delta)
        FFTdata = FFTdata * np.exp(1j * 2. * np.pi * fftfreq * dt)
        FFTdata = FFTdata.astype(np.complex64)
        sf.ifft(FFTdata, n=n, overwrite_x=True)
        trace.data = np.real(FFTdata[:len(trace.data)]).astype(np.float64)
        trace.stats.starttime += dt
        del FFTdata, fftfreq
        return trace
    else:
        return trace