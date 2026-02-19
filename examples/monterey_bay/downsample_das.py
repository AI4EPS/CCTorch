import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import fsspec
import numpy as np
from obspy.signal.interpolation import lanczos_interpolation
import pandas as pd
from args import parse_args
from tqdm import tqdm

import h5py
import torch
from scipy import signal
from datetime import datetime


def preprocess_tensor_batch(x: torch.Tensor,
                            fs: float,
                            target_fs: float | None = None,
                            taper_length: float = 1.5,
                            taper_pct: float = 0,
                            hp_freq: float = 0.01,
                            hp_order: int = 4,
                            lp_freq: float = 25,
                            lp_order: int = 8,
                            lanczos_a: int = 8):

    assert x.ndim == 2, "Expect (batch, time)"
    batch, n = x.shape
    device = x.device

    arr = x.detach().to('cpu').contiguous().numpy().astype(np.float32, copy=False)

    arr -= arr.mean(axis=1, keepdims=True)
    arr = signal.detrend(arr, axis=1, type="linear")

    taper_pct = (taper_length*fs) / n
    alpha = max(0.0, min(1.0, 2.0 * taper_pct))
    if alpha > 0:
        win = signal.windows.tukey(n, alpha=alpha, sym=True).astype(np.float32)
        arr *= win[None, :]

    nyq = 0.5 * fs
    if hp_freq is not None and hp_freq > 0:
        sos_hp = signal.butter(hp_order, hp_freq / nyq, btype='highpass', output='sos')
        arr = signal.sosfiltfilt(sos_hp, arr, axis=1, padtype='odd', padlen=min(3*n-1, 150))
    if lp_freq is not None and lp_freq < nyq:
        sos_lp = signal.butter(lp_order, lp_freq / nyq, btype='lowpass', output='sos')
        arr = signal.sosfiltfilt(sos_lp, arr, axis=1, padtype='odd', padlen=min(3*n-1, 150))

    if target_fs is not None and target_fs != fs:
        try:
            t_old = np.arange(n, dtype=np.float64) / fs
            n_new = int(round(n * (target_fs / fs)))
            t_new = np.arange(n_new, dtype=np.float64) / target_fs
            out = np.empty((batch, n_new), dtype=np.float32)
            for i in range(batch):
                out[i] = lanczos_interpolation(arr[i].astype(np.float64),
                                               t_old, t_new, a=lanczos_a).astype(np.float32)
            arr = out
            fs = target_fs
        except Exception:
            from math import gcd
            up = int(round(target_fs))
            down = int(round(fs))
            g = gcd(up, down)
            up //= g; down //= g
            arr = signal.resample_poly(arr, up, down, axis=1).astype(np.float32, copy=False)
            fs = target_fs
        n = arr.shape[1]

    arr -= arr.mean(axis=1, keepdims=True)

    taper_pct = 0.04
    alpha = max(0.0, min(1.0, 2.0 * taper_pct))
    if alpha > 0:
        win = signal.windows.tukey(n, alpha=alpha, sym=True).astype(np.float32)
        arr *= win[None, :]

    out = torch.from_numpy(arr).to(device=device)
    return out, fs

    
def save_das_h5(root_path, result_path, result_file, data, raw_dtype, new_sampling_rate, config):
    protocol = config["protocol_save"]
    token_file = config["token_file"]
    bucket = config["bucket"]
    downsample_type = config["downsample_type"]

    fs = fsspec.filesystem(protocol, token=token_file)
    waveforms_dir = "waveforms_das"

    out_path_local = f"{root_path}{waveforms_dir}/{result_path}/{downsample_type}"
    if not os.path.exists(out_path_local):
        print(f"Creating local directory: {out_path_local}")
        os.makedirs(out_path_local, exist_ok=True)

    out_path_cloud = f"{bucket}/{waveforms_dir}/{result_path}/{downsample_type}"
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
        acq_grp.attrs["sampling_rate_hz"] = float(new_sampling_rate)

    if protocol != "file":
        print(f"Uploading {out_path_local}/{result_file} to {out_path_cloud}/{result_file}")
        fs.put(f"{out_path_local}/{result_file}", f"{out_path_cloud}/{result_file}")
        os.remove(f"{out_path_local}/{result_file}")

def downsample_h5(fname, raw_freq=200, target_freq=20, root_path="./", config=None, dataset_keys=[]):
    protocol = config["protocol_load"]
    token_file = config["token_file_load"]
    downsample_type = config["downsample_type"]
    
    fs = fsspec.filesystem(protocol, token=token_file)

    all_data_parts = []
    # try:
    for i, tmp in enumerate(fname.split("|")):
        print(f"Loading file #{i}: {tmp}")
        with fs.open(tmp, "rb") as f:
            with h5py.File(f, "r") as hf:
                meta = hf["Acquisition/Raw[0]/RawData"][:]
                raw_dtype = hf["Acquisition/Raw[0]/RawData"].dtype
                meta = torch.from_numpy(meta)
                if downsample_type == 'raw':
                    meta = meta.T.contiguous()
                else:
                    meta = meta.contiguous()
        if i == 0:
            if downsample_type == 'raw':
                date_time_start = tmp.split("_")[-1].split("Z")[0]
            else:
                date_time_start = tmp.split("_")[-2]
            nw_st = tmp.split("/")[-1].split(date_time_start)[0]
            
        all_data_parts.append(meta)
        date_time_end = tmp.split("_")[-1].split("Z")[0]

        info = {}
        for key in dataset_keys:
            info[key] = f[key][:]

    data = torch.cat(all_data_parts, dim=1)
    data, new_fs = preprocess_tensor_batch(data, fs=raw_freq, target_fs=target_freq)
    # except Exception as e:
    #     print(f"Error reading {fname}:\n{e}")
    #     return None

    result_file = f"{nw_st}{date_time_start}_{date_time_end}Z.h5"
    print(f"Saving concatenated file to: {result_file}")
    date_time = datetime.strptime(date_time_start, "%Y-%m-%dT%H%M%S")
    year, jday = date_time.timetuple().tm_year, date_time.timetuple().tm_yday
    
    results = data
    result_path = f'{nw_st.split("_")[0]}/{year:04d}/{jday:03d}'
    save_das_h5(root_path, result_path, result_file, results, raw_dtype, new_fs, config)


if __name__ == "__main__":

    args = parse_args()
    year = f"{args.year:04d}"
    jday = f"{args.jday:03d}"
    hr = f"{args.hr:02d}"
    minute = f"{args.minute:02d}"
    root_path = args.root_path
    protocol_load = args.protocol_load
    protocol_save = args.protocol_save
    token_file = args.token_file
    token_file_load = args.token_file_load
    token_file_save = args.token_file_save
    bucket = args.bucket_das

    downsample_folder = args.downsample_folder
    downsample_type = args.downsample_type

    raw_freq = args.raw_freq
    target_freq = args.target_freq

    # fs = fsspec.filesystem(protocol_load, token=token_file)
    # with fs.open(f"{bucket}/DASh5_list/DASh5_1_{year}_{jday}_{hr}_{minute}.txt", "r") as f:
    #     h5s = pd.read_csv(f)["file_name"].tolist()

    # fs_local = fsspec.filesystem("file")
    fs_local = fsspec.filesystem(protocol_save, token=None)
    # local_path = "../../scripts/das_preprocess/"
    local_path = f"{bucket}/das_preprocess/"

    print(f"Reading preprocess file list from {local_path}...")
    with fs_local.open(f"{local_path}{downsample_folder}/mbdas_h5_down_{year}_{jday}_{hr}_{minute}.txt", "r") as f:
        h5s = pd.read_csv(f)["file_name"].tolist()

    config = {
        "protocol_load": protocol_load,
        "protocol_save": protocol_save,
        "token_file": token_file,
        "token_file_load": token_file_load,
        "token_file_save": token_file_save,
        "bucket": bucket,
        "downsample_type": downsample_type,
    }

    num_workers = os.cpu_count()
    print(f"Processing {len(h5s)} files using {num_workers} workers")

    # with ThreadPoolExecutor(max_workers=num_workers) as executor:
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for h5 in h5s:
            future = executor.submit(
                downsample_h5,
                h5,
                raw_freq=raw_freq,
                target_freq=target_freq,
                config=config,
            )
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downsampling"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {e}")
