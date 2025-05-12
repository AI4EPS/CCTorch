# %%
import concurrent.futures
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
import obspy
import pandas as pd
from tqdm import tqdm
from args import parse_args


args = parse_args()

# %%
fs = fsspec.filesystem("gs", anon=True)


# %%
def read_inventory_file(args):
    root, file = args
    if file.endswith(".xml"):
        with fs.open(f"{root}/{file}", "rb") as f:
            return obspy.read_inventory(f, level="response")
    return None


file_args = []
for root, dirs, files in fs.walk("gs://quakeflow_share/California/FDSNstationXML", maxdepth=2):
    file_args.extend([(root, file) for file in files])

inventory = obspy.Inventory()
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(read_inventory_file, args) for args in file_args]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Reading XML files"):
        inv = future.result()
        if inv is not None:
            inventory += inv

# %%
station_dict = defaultdict(dict)
for network in tqdm(inventory, desc="Writing inventory"):
    for station in network:
        for channel in station:
            sid = f"{network.code}.{station.code}.{channel.location_code}.{channel.code}"
            station_dict[sid] = {
                "network": network.code,
                "station": station.code,
                "location": channel.location_code,
                "channel": channel.code,
                "longitude": channel.longitude,
                "latitude": channel.latitude,
                "elevation_m": channel.elevation,
                "sensitivity": (
                    round(channel.response.instrument_sensitivity.value, 2)
                    if channel.response and channel.response.instrument_sensitivity
                    else None
                ),
            }

# %%
stations = pd.DataFrame.from_dict(station_dict, orient="index")
stations.to_csv(f"stations.csv", index=False)


# %%
protocol = args.protocol
token_file = args.token_file
bucket = args.bucket

fs = fsspec.filesystem(protocol, token=token_file)
fs.put(f"stations.csv", f"{bucket}/ambient_noise/network/stations.csv")
