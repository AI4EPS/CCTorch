import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import fsspec
import pandas as pd
from args import parse_args
from tqdm import tqdm

import h5py
import torch
from datetime import datetime
import numpy as np
    
def save_downchannel_das_h5(root_path, result_path, result_file, data, raw_dtype, channel_lst, config, group="Acquisition/Raw[0]"):
    protocol = config["protocol_save"]
    token_file = config["token_file"]
    bucket = config["bucket"]

    fs = fsspec.filesystem(protocol, token=token_file)
    waveforms_dir = "waveforms_das"

    out_path_local = f"{root_path}{waveforms_dir}/{result_path}"
    if not os.path.exists(out_path_local):
        print(f"Creating local directory: {out_path_local}")
        os.makedirs(out_path_local, exist_ok=True)

    out_path_cloud = f"{bucket}/{waveforms_dir}/{result_path}"
    if not os.path.exists(out_path_cloud):
        print(f"Creating cloud directory: {out_path_cloud}")
        os.makedirs(out_path_cloud, exist_ok=True)

    n_ch, n_samp_total = data.shape

    print(f"Saving fused DAS HDF5 to local path: {out_path_local}")
    with h5py.File(f"{out_path_local}/{result_file}", "w") as h_out:
        acq_grp   = h_out.require_group("Acquisition")
        raw_grp   = acq_grp.require_group("Raw[0]")

        d_raw = raw_grp.create_dataset(
            "RawData",
            shape=(n_ch, n_samp_total),
            maxshape=(n_ch, None),
            dtype=raw_dtype,
            chunks=(min(n_ch, 256), 65536),  # tune chunks as needed
            compression="gzip",
            compression_opts=4,
        )
        d_raw[:] = data
        acq_grp.attrs["channel_lst"] = channel_lst

    if protocol != "file":
        print(f"Uploading {out_path_local}/{result_file} to {out_path_cloud}/{result_file}")
        fs.put(f"{out_path_local}/{result_file}", f"{out_path_cloud}/{result_file}")
        os.remove(f"{out_path_local}/{result_file}")

def down_channel(data, channel_lst):
    data_down = data[channel_lst, :]
    return data_down


def downchannel_h5(fname, channel_lst=[], root_path="./", config=None, dataset_keys=[]):
    protocol = config["protocol_load"]
    token_file = config["token_file_load"]
    
    fs = fsspec.filesystem(protocol, token=token_file)


    all_data_parts = []
    try:
        for i, tmp in enumerate(fname.split("|")):
            print(f"Loading file #{i}: {tmp}")
            with fs.open(tmp, "rb") as f:
                with h5py.File(f, "r") as hf:
                    ds = hf["Acquisition/Raw[0]/RawData"]
                    raw_dtype = ds.dtype
                    meta = ds[:,channel_lst]
                    meta = torch.from_numpy(meta)
                    meta = meta.T.contiguous()
            if i == 0:
                date_time_start = tmp.split("_")[-1].split("Z")[0]
                nw_st = tmp.split("/")[-1].split(date_time_start)[0]

            date_time_end = tmp.split("_")[-1].split("Z")[0]
            all_data_parts.append(meta)

            info = {}
            for key in dataset_keys:
                info[key] = f[key][:]

        data = np.concatenate(all_data_parts, axis=1)

    except Exception as e:
        print(f"Error reading {fname}:\n{e}")
        return None

    result_file = f"{nw_st}{date_time_start}_{date_time_end}Z.h5"
    print(f"Saving concatenated file to: {result_file}")
    date_time = datetime.strptime(date_time_start, "%Y-%m-%dT%H%M%S")
    year, jday = date_time.timetuple().tm_year, date_time.timetuple().tm_yday
    
    result_path = f'{nw_st.split("_")[0]}/{year:04d}/{jday:03d}/downchannel'
    save_downchannel_das_h5(root_path, result_path, result_file, data, raw_dtype, channel_lst, config)


if __name__ == "__main__":
    # --- parse arguments ---
    args = parse_args()
    year = f"{args.year:04d}"
    jday = f"{args.jday:03d}"
    hr = f"{args.hr:02d}"
    minute = f"{args.minute:02d}"
    protocol_load = args.protocol_load
    protocol_save = args.protocol_save
    token_file = args.token_file
    token_file_save = args.token_file_save
    bucket = args.bucket_das

    # --- set up filesystem and paths ---
    # fs_local = fsspec.filesystem("file")
    fs_local = fsspec.filesystem(protocol_save, token=None)
    local_path = f"{bucket}/das_preprocess/"

    # --- read preprocess file list ---
    print(f"Reading preprocess file list from {local_path}...")
    # local_path = "../../scripts/das_preprocess/"
    with fs_local.open(f"{local_path}mbdas_h5_downchannel_list/mbdas_h5_down_{year}_{jday}_{hr}_{minute}.txt", "r") as f:
        h5s = pd.read_csv(f)["file_name"].tolist()

    # --- check if downchannel files already exist, if yes, skip processing ---
    path_gcloud = f'cctorch/ambient_noise_das/waveforms_das/MBARI/{year}/{jday}/downchannel'
    try:
        file_exist = fs_local.ls(path_gcloud)
        file_exist = [f.split("/")[-1] for f in file_exist if f.endswith("Z.h5")]
        h5s_check = h5s[0].split("|")
        file_check = h5s_check[0].split("/")[-1].split(f'Z.h')[0] + '_' + h5s_check[-1].split("/")[-1].split('_')[-1]
        if file_check in file_exist:
            print(f"File already exists in {path_gcloud}, skipping downchannel processing.")
            exit(0)
    except (FileNotFoundError, IndexError) as e:
        print(f"Proceeding: No existing files or path found (Error: {e})")
    

    # --- read channel list ---
    with fs_local.open(f"{local_path}channel_list_2.txt", "r") as f:
        channel_lst = pd.read_csv(f)["channel"].tolist()
        
        date = datetime.strptime(f"{year}_{jday}_{hr}_{minute}", "%Y_%j_%H_%M")
        if date < datetime.strptime(f"2023_117_22_30", "%Y_%j_%H_%M"):
            print(f"Using old channel list for date {date}")
            print(f"Original channel list: {channel_lst}")
            channel_lst = [ch - 55 for ch in channel_lst]
            print(f"Adjusted channel list: {channel_lst}")

    # --- assign config for downchannel processing ---
    config = {
        "protocol_load": protocol_load,
        "protocol_save": protocol_save,
        "token_file": token_file,
        "token_file_load": "token/x-berkeley-mbari-das-8c2333fca1b2.json",
        "token_file_save": token_file_save,
        "bucket": bucket,
    }

    # --- process files in parallel ---
    num_workers = os.cpu_count()
    print(f"Processing {len(h5s)} files using {num_workers} workers")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for h5 in h5s:
            future = executor.submit(
                downchannel_h5,
                h5,
                channel_lst=channel_lst,
                config=config,
            )
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downsampling"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {e}")
