# %%
import math
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import gamma
import h5py
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import scipy
import scipy.signal
from tqdm import tqdm


# %%
@dataclass
class Config:
    sampling_rate: int = 100
    time_before: float = 2
    time_after: float = 2
    component: str = "ENZ123"
    degree2km: float = 111.2

    def __init__(self) -> None:
        self.nt = int((self.time_before + self.time_after) * self.sampling_rate)
        pass


# %%
def resample(data, sampling_rate, new_sampling_rate):
    """
    data is a 1D numpy array
    implement resampling using numpy
    """
    if sampling_rate == new_sampling_rate:
        return data
    else:
        # resample
        n = data.shape[0]
        t = np.linspace(0, 1, n)
        t_interp = np.linspace(0, 1, int(n * new_sampling_rate / sampling_rate))
        data_interp = np.interp(t_interp, t, data)
        return data_interp


def detrend(data):
    """
    data is a 1D numpy array
    implement detrending using scipy to remove a linear trend
    """
    return scipy.signal.detrend(data, type="linear")


def taper(data, taper_type="hann", taper_fraction=0.05):
    """
    data is a 1D numpy array
    implement tapering using scipy
    """
    if taper_type == "hann":
        taper = scipy.signal.hann(int(data.shape[0] * taper_fraction))
    elif taper_type == "hamming":
        taper = scipy.signal.hamming(int(data.shape[0] * taper_fraction))
    elif taper_type == "blackman":
        taper = scipy.signal.blackman(int(data.shape[0] * taper_fraction))
    else:
        raise ValueError("Unknown taper type")
    taper = taper[: len(taper) // 2]
    taper = np.hstack((taper, np.ones(data.shape[0] - taper.shape[0] * 2), taper[::-1]))
    return data * taper


def filter(data, type="highpass", freq=1.0, sampling_rate=100.0):
    """
    data is a 1D numpy array
    implement filtering using scipy
    """
    if type == "highpass":
        b, a = scipy.signal.butter(2, freq, btype="highpass", fs=sampling_rate)
    elif type == "lowpass":
        b, a = scipy.signal.butter(2, freq, btype="lowpass", fs=sampling_rate)
    elif type == "bandpass":
        b, a = scipy.signal.butter(2, freq, btype="bandpass", fs=sampling_rate)
    elif type == "bandstop":
        b, a = scipy.signal.butter(2, freq, btype="bandstop", fs=sampling_rate)
    else:
        raise ValueError("Unknown filter type")
    return scipy.signal.filtfilt(b, a, data)


# %%
def extract_template(year_dir, jday, events, stations, picks, config, mseed_path, output_path, figure_path):

    # %%
    waveforms_dict = {}
    for station_id in tqdm(stations["station_id"], desc=f"Loading: "):
        net, sta, loc, chn = station_id.split(".")
        key = f"{net}.{sta}.{chn}[{config.component}].mseed"
        try:
            stream = obspy.read(jday / key)
            stream.merge(method=1, interpolation_samples=0)
            waveforms_dict[key] = stream
        except Exception as e:
            print(e)
            continue

    # %%
    picks["station_component_index"] = picks.apply(lambda x: f"{x.station_id}.{x.phase_type}", axis=1)

    # %%
    with h5py.File(output_path / f"{year_dir.name}-{jday.name}.h5", "w") as fp:

        begin_time = datetime.strptime(f"{year_dir.name}-{jday.name}", "%Y-%j").replace(tzinfo=timezone.utc)
        end_time = begin_time + timedelta(days=1)
        events_ = events[(events["event_time"] > begin_time) & (events["event_time"] < end_time)]

        num_event = 0
        for event_index in tqdm(events_["event_index"], desc=f"Cutting event {year_dir.name}-{jday.name}.h5"):

            picks_ = picks.loc[event_index]
            picks_ = picks_.set_index("station_component_index")

            event_loc = events_.loc[event_index][["x_km", "y_km", "z_km"]].to_numpy().astype(np.float32)
            event_loc = np.hstack((event_loc, [0]))[np.newaxis, :]
            station_loc = stations[["x_km", "y_km", "z_km"]].to_numpy()

            h5_event = fp.create_group(f"{event_index}")

            for i, phase_type in enumerate(["P", "S"]):

                travel_time = gamma.seismic_ops.calc_time(
                    event_loc,
                    station_loc,
                    [phase_type.lower() for _ in range(len(station_loc))],
                ).squeeze()

                predicted_phase_timestamp = events_.loc[event_index]["event_timestamp"] + travel_time
                # predicted_phase_time = [events_.loc[event_index]["event_time"] + pd.Timedelta(seconds=x) for x in travel_time]

                for c in config.component:

                    h5_template = h5_event.create_group(f"{phase_type}_{c}")

                    data = np.zeros((len(stations), config.nt))
                    label = []
                    snr = []
                    empty_data = True

                    # fig, axis = plt.subplots(1, 1, squeeze=False, figsize=(6, 10))
                    for j, station_id in enumerate(stations["station_id"]):

                        if f"{station_id}_{phase_type}" in picks_.index:
                            ## TODO: check if multiple phases for the same station
                            phase_timestamp = picks_.loc[f"{station_id}_{phase_type}"]["phase_timestamp"]
                            predicted_phase_timestamp[j] = phase_timestamp
                            label.append(1)
                        else:
                            label.append(0)

                        net, sta, loc, chn = station_id.split(".")
                        key = f"{net}.{sta}.{chn}[{config.component}].mseed"

                        if key in waveforms_dict:

                            trace = waveforms_dict[key]
                            trace = trace.select(channel=f"*{c}")
                            if len(trace) == 0:
                                continue
                            if len(trace) > 1:
                                print(f"More than one trace: {trace}")
                            trace = trace[0]

                            begin_time = (
                                predicted_phase_timestamp[j]
                                - trace.stats.starttime.datetime.replace(tzinfo=timezone.utc).timestamp()
                                - config.time_before
                            )
                            end_time = (
                                predicted_phase_timestamp[j]
                                - trace.stats.starttime.datetime.replace(tzinfo=timezone.utc).timestamp()
                                + config.time_after
                            )

                            trace_data = trace.data[
                                int(begin_time * trace.stats.sampling_rate) : int(end_time * trace.stats.sampling_rate)
                            ].astype(np.float32)
                            if len(trace_data) < (config.nt // 2):
                                continue
                            std = np.std(trace_data)
                            if std == 0:
                                continue

                            if trace.stats.sampling_rate != config.sampling_rate:
                                # print(f"Resampling {trace.id}: {trace.stats.sampling_rate}Hz -> {config.sampling_rate}Hz")
                                trace_data = resample(trace_data, trace.stats.sampling_rate, config.sampling_rate)

                            trace_data = detrend(trace_data)
                            trace_data = taper(trace_data, taper_type="hann", taper_fraction=0.05)
                            trace_data = filter(trace_data, type="highpass", freq=1, sampling_rate=config.sampling_rate)

                            empty_data = False
                            data[j, : config.nt] = trace_data[: config.nt]
                            snr.append(np.std(trace_data[config.nt // 2 :]) / np.std(trace_data[: config.nt // 2]))

                    #         # axis[0, 0].plot(
                    #         #     np.arange(len(trace_data)) / config.sampling_rate - config.time_before,
                    #         #     trace_data / std / 3.0 + j,
                    #         #     c="k",
                    #         #     linewidth=0.5,
                    #         #     label=station_id,
                    #         # )

                    if not empty_data:
                        data = np.array(data)
                        data_ds = h5_template.create_dataset("data", data=data, dtype=np.float32)
                        data_ds.attrs["nx"] = data.shape[0]
                        data_ds.attrs["nt"] = data.shape[1]
                        data_ds.attrs["dt_s"] = 1.0 / config.sampling_rate
                        data_ds.attrs["time_before_s"] = config.time_before
                        data_ds.attrs["time_after_s"] = config.time_after
                        tt_ds = h5_template.create_dataset("travel_time", data=travel_time, dtype=np.float32)
                        tti_ds = h5_template.create_dataset(
                            "travel_time_index", data=np.round(travel_time * config.sampling_rate), dtype=np.int32
                        )
                        ttt_ds = h5_template.create_dataset("travel_time_type", data=label, dtype=np.int32)
                        ttt_ds.attrs["label"] = ["predicted", "auto_picks", "manual_picks"]
                        sta_ds = h5_template.create_dataset(
                            "station_id",
                            data=stations["station_id"].to_numpy(),
                            dtype=h5py.string_dtype(encoding="utf-8"),
                        )
                        snr_ds = h5_template.create_dataset("snr", data=snr, dtype=np.float32)

                    # if has_data:
                    #     fig.savefig(figure_path / f"{event_index}_{phase_type}_{c}.png")
                    #     plt.close(fig)

                num_event += 1
                if num_event > 20:
                    break


# %%
if __name__ == "__main__":

    # %%
    config = Config()

    min_longitude, max_longitude, min_latitude, max_latitude = [34.7 + 0.4, 39.7 - 0.4, 35.5, 39.5 - 0.1]
    center = [(min_longitude + max_longitude) / 2, (min_latitude + max_latitude) / 2]
    config.center = center
    config.xlim_degree = [min_longitude, max_longitude]
    config.ylim_degree = [min_latitude, max_latitude]

    stations = pd.read_json("../../EikoLoc/stations.json", orient="index")
    stations["station_id"] = stations.index
    stations = stations[
        (stations["longitude"] > config.xlim_degree[0])
        & (stations["longitude"] < config.xlim_degree[1])
        & (stations["latitude"] > config.ylim_degree[0])
        & (stations["latitude"] < config.ylim_degree[1])
    ]
    # stations["distance_km"] = stations.apply(
    #     lambda x: math.sqrt((x.latitude - config.center[1]) ** 2 + (x.longitude - config.center[0]) ** 2)
    #     * config.degree2km,
    #     axis=1,
    # )
    # stations.sort_values(by="distance_km", inplace=True)
    # stations.drop(columns=["distance_km"], inplace=True)
    # stations.sort_values(by="latitude", inplace=True)
    stations["x_km"] = stations.apply(
        lambda x: (x.longitude - config.center[0]) * np.cos(np.deg2rad(config.center[1])) * config.degree2km, axis=1
    )
    stations["y_km"] = stations.apply(lambda x: (x.latitude - config.center[1]) * config.degree2km, axis=1)
    stations["z_km"] = stations.apply(lambda x: -x.elevation_m / 1e3, axis=1)

    # %%
    events = pd.read_csv(
        "../../EikoLoc/eikoloc_catalog.csv", parse_dates=["time"], date_parser=lambda x: pd.to_datetime(x, utc=True)
    )
    events = events[events["time"].notna()]
    events.sort_values(by="time", inplace=True)
    events.rename(columns={"time": "event_time"}, inplace=True)
    events["event_timestamp"] = events["event_time"].apply(lambda x: x.timestamp())
    events["x_km"] = events.apply(
        lambda x: (x.longitude - config.center[0]) * np.cos(np.deg2rad(config.center[1])) * config.degree2km, axis=1
    )
    events["y_km"] = events.apply(lambda x: (x.latitude - config.center[1]) * config.degree2km, axis=1)
    events["z_km"] = events.apply(lambda x: x.depth_km, axis=1)
    event_index = list(events["event_index"])

    # %%
    picks = pd.read_csv(
        "../../EikoLoc/gamma_picks.csv", parse_dates=["phase_time"], date_parser=lambda x: pd.to_datetime(x, utc=True)
    )
    picks = picks[picks["event_index"] != -1]
    picks["phase_timestamp"] = picks["phase_time"].apply(lambda x: x.timestamp())
    picks = picks.merge(stations, on="station_id")
    picks = picks.merge(events, on="event_index", suffixes=("_station", "_event"))

    # %%
    events["index"] = events["event_index"]
    events = events.set_index("index")
    picks["index"] = picks["event_index"]
    picks = picks.set_index("index")

    # %%
    mseed_path = Path("../../convert_format/wf/")
    figure_path = Path("./figures/")
    output_path = Path("./templates/")
    if not figure_path.exists():
        figure_path.mkdir()
    if not output_path.exists():
        output_path.mkdir()

    # %%
    ncpu = mp.cpu_count()
    with mp.Pool(ncpu) as pool:
        pool.starmap(
            extract_template,
            [
                (
                    year_dir,
                    jday,
                    events,
                    stations,
                    picks,
                    config,
                    mseed_path,
                    output_path,
                    figure_path,
                )
                for year_dir in mseed_path.iterdir()
                for jday in year_dir.iterdir()
            ],
        )

# %%
