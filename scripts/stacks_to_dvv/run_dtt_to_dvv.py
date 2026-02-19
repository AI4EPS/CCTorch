import gcsfs
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
from obspy.signal.regression import linear_regression
from concurrent.futures import ProcessPoolExecutor, as_completed

from args import parse_args
import importlib

args = parse_args()

def scan_avai_pair_date_cloud(pair_dir):
    fs = gcsfs.GCSFileSystem()
    if pair_dir.startswith("gs://"):
        prefix = pair_dir
    else:
        prefix = f"gs://{pair_dir}"
    try:
        file_date_lst = fs.ls(prefix)
    except Exception as e:
        print(f"Failed to list {pair_dir}: {e}")
        return []
    date_lst = []
    for mwcs_file in file_date_lst:
        try:
            date_str = mwcs_file.split('/')[-1].split('.')[0]
            date_lst.append(date_str)
        except Exception as e:
            print(f"Failed to process {mwcs_file}: {e}")
            continue

    print(f"{prefix}: Found {len(date_lst)} dates")
    return sorted(list(set(date_lst)))


def process_pair(pair_dir: str, config, comp: str):
    file_dvv_out = config["file_dvv_out"]
    mov_stacks = config["mov_stacks"]
    filterid = config["filterid"]
    time_type = config["time_type"]

    pair_name = pair_dir.split('/')[-1]

    date_available = scan_avai_pair_date_cloud(pair_dir)
    if not date_available:
        print(f"No available dates for pair_dir: {pair_dir}, skipping.")
        return
    
    fs_local = gcsfs.GCSFileSystem()
    if not fs_local.exists(file_dvv_out):
            fs_local.makedirs(file_dvv_out, exist_ok=True)
    
    comp_lst_dtt = [comp]

    Dates = []
    dvv_M_lst = []
    dvv_M0_lst = []
    for components in comp_lst_dtt:
        for current in date_available:
            for mov_stack in mov_stacks:
                if pair_dir.startswith("gs://"):
                    day = os.path.join(pair_dir, f"{current}.txt")
                else:
                    day = os.path.join("gs://", pair_dir, f"{current}.txt")

                with fs_local.open(day, 'r') as fr:
                    lines = fr.readlines()
                    if len(lines) < 2:
                        print(f"File {day} is empty or has no valid data, skipping.")
                        continue
                    line = lines[1]
                    date = line.split(',')[0]
                    value_M = -1*float(line.split(',')[2])
                    value_M0 = -1*float(line.split(',')[6])
                    dvv_M_lst.append(value_M)
                    dvv_M0_lst.append(value_M0)
                    Dates.append(date)

    output = os.path.join(file_dvv_out,
                        'DVV', "%02i" % filterid, f"%03i_{time_type}" % mov_stack,
                        components)
    
    df_out = pd.DataFrame(
                        {'dvv_M': dvv_M_lst, 'dvv_M0': dvv_M0_lst},
                        index=Dates)

    if not fs_local.exists(output):
                        fs_local.makedirs(output, exist_ok=True)
    fn = os.path.join(output, '%s.txt' % pair_name)
    if fs_local.exists(fn):
        with fs_local.open(fn, 'rt') as f_in:
            existing = pd.read_csv(f_in, index_col="dvv_M", parse_dates=True)
        
        for _, row in df_out.iterrows():
            if row.dvv_M in existing.index.values:
                existing.drop(row.dvv_M, inplace=True)
                # logger.debug("Pair: %s is already in the output file, overwriting" % row.Pairs)
        existing["dvv_M"] = existing.index.values
        existing.set_index("Date", inplace=True)
        df_out = pd.concat([df_out, existing])
                        
    with fs_local.open(fn, 'wt') as gcs_file:
        df_out.to_csv(gcs_file, index_label='Date')

    del df_out, value_M, value_M0, Dates
    del output


if __name__ == "__main__":
    node_rank = args.node_rank
    num_nodes = args.num_nodes

    cfg = importlib.import_module(f"configs.{args.project}")
    component_mwcs = cfg.component_mwcs
    # file_list_path = cfg.dtt_paths
    folder_dtt = cfg.folder_dtt

    config = {"file_dvv_out": cfg.folder_dvv,
              "mov_stacks": cfg.mov_stacks,
              "filterid": cfg.filterid_dtt,
              "time_type": cfg.time_type}

    fs = gcsfs.GCSFileSystem()
    fs.invalidate_cache("gs://")

    path_dtt = f'{folder_dtt}DTT/01/001_{cfg.time_type}/{component_mwcs}'
    pair_dirs = fs.ls(path_dtt)


    # with open(file_list_path, "r") as f:
    #     pair_dirs = [line.strip() for line in f.readlines()]
    print(f"Total pair_dirs found: {len(pair_dirs)} from {path_dtt}")

    pair_dirs = pair_dirs[node_rank::num_nodes]

    max_workers = min(len(pair_dirs), os.cpu_count() or 4)
    print(f"Using {max_workers} workers for {len(pair_dirs)} pair_dirs")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pair, p, config, component_mwcs): p for p in pair_dirs}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"Error in pair_dir {p}: {e}")

    print(f"Node rank {node_rank} completed processing {len(pair_dirs)} pair_dirs.")