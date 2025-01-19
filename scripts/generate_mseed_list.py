# %%
import fsspec
import h5py
import obspy
import pandas as pd


def map_cloud_path(root_path, provider, starttime, network, station, location, channels):
    paths = []
    for channel in channels.split(","):
        if isinstance(starttime, str):
            starttime = pd.Timestamp(starttime)
        if provider.lower() == "scedc":
            year = starttime.strftime("%Y")
            dayofyear = starttime.strftime("%j")
            if location == "":
                location = "__"
            path = f"{root_path}/{provider.lower()}-pds/continuous_waveforms/{year}/{year}_{dayofyear}/{network}{station:_<5}{channel}{location:_<2}_{year}{dayofyear}.ms"
        elif provider.lower() == "ncedc":
            year = starttime.strftime("%Y")
            dayofyear = starttime.strftime("%j")
            path = f"{root_path}/{provider.lower()}-pds/continuous_waveforms/{network}/{year}/{year}.{dayofyear}/{station}.{network}.{channel}.{location}.D.{year}.{dayofyear}"
        else:
            raise ValueError(f"Unknown provider: {provider}")
        paths.append(path)

    return paths


# %%
if __name__ == "__main__":
    # %%
    mseed_list = [
        {
            "provider": "ncedc",
            "network": "NC",
            "station": "KCT",
            "location": "",
            "channels": "HHE,HHN,HHZ",
            "year": "2012",
            "month": "01",
            "day": "01",
        },
        {
            "provider": "ncedc",
            "network": "NC",
            "station": "KRP",
            "location": "",
            "channels": "HHE,HHN,HHZ",
            "year": "2012",
            "month": "01",
            "day": "01",
        },
        {
            "provider": "ncedc",
            "network": "NC",
            "station": "KHMB",
            "location": "",
            "channels": "HHE,HHN,HHZ",
            "year": "2012",
            "month": "01",
            "day": "01",
        },
    ]
    # %%
    file_list = []
    root_path = "s3:/"
    for mseed_info in mseed_list:
        starttime = pd.Timestamp(f"{mseed_info['year']}-{mseed_info['month']}-{mseed_info['day']}T00:00:00")
        file_path = map_cloud_path(
            root_path,
            mseed_info["provider"],
            starttime,
            mseed_info["network"],
            mseed_info["station"],
            mseed_info["location"],
            mseed_info["channels"],
        )
        file_list.append("|".join(file_path))
        # with fsspec.open(file_path, "rb", anon=True) as f:
        #     stream = obspy.read(f)
        #     stream.plot()  # %%

    with open("data_list.txt", "w") as f:
        f.write("file_name\n")
        f.write("\n".join(file_list))

    num_files = len(file_list)
    with open("pair_list.txt", "w") as f:
        for i in range(num_files):
            for j in range(i + 1, num_files):
                f.write(f"{i},{j}\n")


# %%
