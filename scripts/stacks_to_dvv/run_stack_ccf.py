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

def load_ccf_all_cloud(st1, st2, f_low, f_high, ref_start, ref_end, ccf_date_lst, sampling_rate=20, component="ZZ"):
    comp_idx = {"EE":0, "EN":1, "EZ":2, "NE":3, "NN":4, "NZ":5, "ZE":6, "ZN":7, "ZZ":8}

    available_date_lst = []
    ccf_filt_lst = []
    ccf_ref_filt_lst = []
    
    for ccf_date in ccf_date_lst:
        year = ccf_date.year
        jday = ccf_date.dayofyear
        file = f'cctorch/ambient_noise/ccf/{year}/{year}.{jday:03d}.zarr/'
        try:
            date = f"{year}.{jday}"
            date_int = int(datetime.strptime(date, "%Y.%j").strftime("%Y%m%d"))

            z_temp = zarr.open(f'gs://{file}', mode='r')
            st1_lst = z_temp['id1'][:]
            st2_lst = z_temp['id2'][:]
            idx = np.where((st1_lst == st1) & (st2_lst == st2))[0]
            
            ccf = z_temp['xcorr'][idx][comp_idx[component]]
            ccf_filt = obspy_filter(ccf, sampling_rate=sampling_rate, f_low=f_low, f_high=f_high)
            ccf_filt_lst.append(ccf_filt)
            if (datetime.strptime(date, "%Y.%j") >= ref_start) and (datetime.strptime(date, "%Y.%j") <= ref_end):
                ccf_ref_filt_lst.append(ccf_filt)

            available_date_lst.append(date_int)
            
        except:
            print(f"{year}.{jday} missing data at {st1}-{st2}.")

    if len(ccf_filt_lst) == 0:
        return None

    return available_date_lst, np.asarray(ccf_filt_lst), np.asarray(ccf_ref_filt_lst)
    

def sliding_stack_ccf(date_lst, ccf_all, half_window_days=2, min_traces=1, stack_method="mean"):
    """
    date_lst: list-like of datetime/date-like objects
    ccf_all:  2D array (N_days, N_lags)
    half_window_days: e.g. 2 for ±2 days
    min_traces: minimum number of CCFs required to form a stack
    stack_method: "mean" or "median"
    """
    date_lst = [datetime.strptime(str(date), "%Y%m%d") for date in date_lst]

    dates = pd.to_datetime(date_lst)
    dates_np = dates.to_numpy().astype("datetime64[D]")
    
    ccf_all = np.asarray(ccf_all)
    n_days, n_lags = ccf_all.shape

    stacked = np.zeros_like(ccf_all)
    valid_mask = np.zeros(n_days, dtype=bool)

    for i in range(n_days):
        center = dates_np[i]
        start = center - np.timedelta64(half_window_days, "D")
        end   = center + np.timedelta64(half_window_days, "D")

        # pick all rows within [center - half_window, center + half_window]
        win_mask = (dates_np >= start) & (dates_np <= end)
        idx = np.where(win_mask)[0]

        if len(idx) < min_traces:
            # not enough data → mark as invalid and maybe keep original or NaN
            stacked[i, :] = np.nan
            continue

        if stack_method == "mean":
            stacked[i, :] = np.mean(ccf_all[idx, :], axis=0)
        elif stack_method == "median":
            stacked[i, :] = np.median(ccf_all[idx, :], axis=0)
        else:
            raise ValueError("stack_method must be 'mean' or 'median'")

        valid_mask[i] = True

    return stacked, valid_mask

def save_stacked_ccf(date, ccf_all, ccf_ref, store_path):
    # --- Convert lists to NumPy arrays ---
    date_array = np.asarray(date)
    ccf_array = np.asarray(ccf_all)
    ccf_ref_array = np.asarray(ccf_ref)

    # -------------------------------------
    print(f"Save stacked result on {store_path}")
    fp = zarr.storage.FsspecStore.from_url(
            store_path, read_only=False, storage_options={"token": "google_default"}
        )
    
    zarr.create_array(fp, name="date", data=date_array, overwrite=True)
    zarr.create_array(fp, name="data", data=ccf_array, overwrite=True)
    zarr.create_array(fp, name="reference", data=ccf_ref_array, overwrite=True)

# ---------- define worker for one pair ----------
def process_pair(pair, config, filterid, ccf_date_lst, component="ZZ"):
    ref_start = datetime.strptime(config.ref_start, "%Y-%m-%d")
    ref_end = datetime.strptime(config.ref_end, "%Y-%m-%d")
    sampling_rate = config.sampling_rate
    half_window_days = config.half_window

    st1, st2 = pair
    # try:
    print(f"[{st1}-{st2}] Start")

    available_date_lst, ccf_filt_lst, ccf_ref_filt_lst = load_ccf_all_cloud(st1, st2, config.f_low_dic[filterid], config.f_high_dic[filterid], ref_start, ref_end, ccf_date_lst, sampling_rate, component)


    if len(available_date_lst) == 0:
        return f"[{st1}-{st2}] no data, skipped"

    ccf_ref = np.nanmean(ccf_ref_filt_lst, axis=0)
    stacked_ccf, valid_mask = sliding_stack_ccf(available_date_lst, 
                                            ccf_filt_lst, 
                                            half_window_days=half_window_days, 
                                            min_traces=1,
                                            stack_method="mean")
    
    available_stacked_date_lst = [d for d, v in zip(available_date_lst, valid_mask) if v]
    ccf_valid = stacked_ccf[valid_mask]

    if len(available_stacked_date_lst) == 0:
        return f"[{st1}-{st2}] no valid stacked days, skipped"
    
    store_path = f'gs://{config.folder_stacks}{filterid:02d}/pm{half_window_days:03d}_DAYS/{component}/{st1}_{st2}.zarr/'
    save_stacked_ccf(available_stacked_date_lst, ccf_valid, ccf_ref, store_path)

    return f"[{st1}-{st2}] done, {len(available_stacked_date_lst)} valid days after stacking"

    # except Exception as e:
    #     return f"[{st1}-{st2}] error: {e}"
    

# ---------- parallel execution ----------
if __name__ == "__main__":
    file_mark = args.file_mark

    # --- import config based on project argument ---
    cfg = importlib.import_module(f"configs.{args.project}")
    
    station_pair_list = cfg.station_pair_list
    if station_pair_list.split('.')[-1] != 'txt':
        station_pair_list += file_mark + '.txt'

    component_list = cfg.component_list

    ccf_date_lst = pd.date_range(
    start=cfg.ccf_start,
    end=cfg.ccf_end,
    freq="D"
    ).to_list()

    config = {"ref_start": cfg.ref_start, 
              "ref_end": cfg.ref_end, 
              "sampling_rate": cfg.sampling_rate, 
              "half_window_days": cfg.half_window_days, 
              "f_low_dic": cfg.f_low_dic, 
              "f_high_dic": cfg.f_high_dic, 
              "folder_stacks": cfg.folder_stacks}

    # ---------- read pair list ----------
    print(f"Load pairs from {station_pair_list}")

    pairs_lst = []
    with open(station_pair_list, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pairs = line.split(' ')[0]
            pairs = pairs.strip()
            st1, st2 = pairs.split('_')
            pairs_lst.append([st1, st2])
            
    num_workers = os.cpu_count()  # tune this based on CPU / I/O / GCS rate limits
    total = len(pairs_lst)

    print(f"Processing {total} station pairs with {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(process_pair, pair, config, filterid, ccf_date_lst, component): pair for pair in pairs_lst for component in component_list for filterid in cfg.filterid_list}

        for i, fut in enumerate(as_completed(futures), start=1):
            msg = fut.result()
            print(f"[{i}/{total}] {msg}")