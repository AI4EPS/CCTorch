import os
import numpy as np
import pandas as pd
import zarr
from obspy.signal.filter import bandpass
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib

from args import parse_args
args = parse_args()

def obspy_filter(data, sampling_rate, f_low, f_high):
    filtered = bandpass(data, freqmin=f_low, freqmax=f_high, df=sampling_rate, corners=4, zerophase=True)
    return filtered

def write_ambient_noise_indexed(xcorr_data, store_path, start_idx, batch_id1=None, batch_id2=None, storage_options=None):
    # Each worker opens its own store connection
    if store_path.startswith("s3://") or store_path.startswith("gs://"):
        store = zarr.storage.FsspecStore.from_url(
            store_path, read_only=False, storage_options=storage_options or {}
        )
    else:
        store = zarr.storage.LocalStore(store_path)

    root = zarr.open_group(store, mode="r+")

    # Write to specific index range - no race condition since ranges don't overlap
    # FIXME: Error writing indices 50110:50174: The object cctorch/ambient_noise/ccf/2025/2025.001.zarr/id1/c/12 exceeded the rate limit for object mutation operations (create, update, and delete). Please reduce your request rate. See https://cloud.google.com/storage/docs/gcs429
    try:
        root["xcorr"][start_idx:start_idx + 1] = xcorr_data
        root["id1"][start_idx:start_idx + 1] = batch_id1
        root["id2"][start_idx:start_idx + 1] = batch_id2
    except Exception as e:
        print(f"Error writing indices {start_idx}:{start_idx + 1}: {e}")



def merge_ccf_all_cloud(start_idx, merge_lst, f_low, f_high, ccf_date, sampling_rate=20, component="ZZ"):
    print(f"Loading {ccf_date} CCFs... with merge_lst={merge_lst} and start_idx={start_idx}")
    year, jday = ccf_date.year, ccf_date.dayofyear
    hour, minute = ccf_date.hour, ccf_date.minute
    file = f'cctorch/ambient_noise_das/ccf_das/{year}/{year}.{jday:03d}.{hour:02d}.{minute:02d}.zarr/'
    # try:
    z_temp = zarr.open(f'gs://{file}', mode='r')
    available_id1 = z_temp['id1'][:]
    available_id2 = z_temp['id2'][:]
    lookup = {(str(id1), str(id2)): i for i, (id1, id2) in enumerate(zip(available_id1, available_id2))}
    ccf_filt_lst = []
    actual_st1_used = []
    actual_st2_used = []
    for pair in merge_lst:
        st1, st2 = pair.split('_')
        idx = lookup.get((st1, st2))
        if idx is not None:
            ccf = z_temp['xcorr'][idx]
            ccf_filt = obspy_filter(ccf, sampling_rate=sampling_rate, f_low=f_low, f_high=f_high)
            ccf_filt_lst.append(ccf_filt)
            actual_st1_used.append(st1)
            actual_st2_used.append(st2)
    if not ccf_filt_lst:
        return f"No CCFs found for pairs on {ccf_date}."
    ccf_merge = np.mean(np.array(ccf_filt_lst), axis=0)
    store_path = file.replace('ccf_das', 'ccf_das_merge')
    batch_id1 = f"{actual_st1_used[0]}_m_{actual_st1_used[-1]}"
    batch_id2 = f"{actual_st2_used[0]}_m_{actual_st2_used[-1]}"
    write_ambient_noise_indexed(ccf_merge, f"gs://{store_path}", start_idx, batch_id1=batch_id1, batch_id2=batch_id2, storage_options=None)
    return f"SUCCESS: Merge {len(ccf_filt_lst)} CCFs at index [{batch_id1}-{batch_id2}] on {ccf_date}"
    # except:
    #     print(f"{year}.{jday:03d}.{hour:02d}.{minute:02d} missing data at {st1}-{st2}.")
    
def process_pair(start_idx, merge_lst, config, filterid, ccf_date, component="ZZ"):
    sampling_rate = config["sampling_rate"]
    f_low = config["f_low_dic"][filterid]
    f_high = config["f_high_dic"][filterid]
    merge_ccf_all_cloud(start_idx, merge_lst, f_low, f_high, ccf_date, sampling_rate, component)
    

# ---------- parallel execution ----------
if __name__ == "__main__":
    file_mark = args.file_mark
    half_window_type = "h"

    # --- import config based on project argument ---
    cfg = importlib.import_module(f"configs.{args.project}")
    
    station_merge_list = cfg.station_merge_list
    print(f"station_merge_list = {station_merge_list}")
    if station_merge_list.split('.')[-1] != 'txt':
        station_merge_list += file_mark + '.txt'
    print(f"station_merge_list after modification = {station_merge_list}")
    

    component_list = cfg.component_list

    ccf_date_lst = pd.date_range(
    start=cfg.ccf_start,
    end=cfg.ccf_end,
    freq=half_window_type
    ).to_list()

    ccf_date_lst = ccf_date_lst[:3]

    config = {"sampling_rate": cfg.sampling_rate, 
              "f_low_dic": cfg.f_low_dic, 
              "f_high_dic": cfg.f_high_dic}

    # ---------- read pair list ----------
    print(f"Load pairs from {station_merge_list}")
    df = pd.read_csv(station_merge_list)
    group_lst = df.columns.tolist()
    group_pair_lst = []
    for group in group_lst:
        pair_lst = df[group].dropna().tolist()
        group_pair_lst.append(pair_lst)
            
    num_workers = os.cpu_count()  # tune this based on CPU / I/O / GCS rate limits
    total = len(group_pair_lst)

    print(f"Processing {total} group pairs with {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(process_pair, start_idx, merge_lst, config, filterid, ccf_date, component_list[0]): (start_idx, merge_lst) for start_idx, merge_lst in enumerate(group_pair_lst) for ccf_date in ccf_date_lst for filterid in cfg.filterid_list}

        for i, fut in enumerate(as_completed(futures), start=1):
            msg = fut.result()
            print(f"[{i}/{total}] {msg}")