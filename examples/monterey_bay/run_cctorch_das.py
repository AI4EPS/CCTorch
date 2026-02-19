# %%
from datetime import datetime
import os

import fsspec
import pandas as pd
from args import parse_args

args = parse_args()


# %%
if __name__ == "__main__":
    local_test_mode = False

    year = args.year
    jday_start = args.jday_start
    jday_end = args.jday_end
    concat_minute = args.concat_minute
    node_rank = args.node_rank
    num_nodes = args.num_nodes
    protocol = args.protocol
    token_file = args.token_file_load
    result_path = args.result_path
    pair_lst = args.pair_lst
    bucket = args.bucket_das

    # get how many days in the year
    jdays = [f'{i:03d}' for i in range(jday_start, jday_end + 1)]
    jdays = jdays[node_rank::num_nodes]

    print(f"{jdays = }")
    # result_path = 'gs://cctorch/ambient_noise_DAS/ccf_DAS_1hr'
    print("Protocol:", protocol, "Token file:", token_file, "result_path:", result_path)

    fs = fsspec.filesystem(protocol, token=token_file)
    
    if local_test_mode:
        hr_lst = [f"{i:02d}" for i in range(0, 1)]
        minutes = [f"{i:02d}" for i in range(0, 5, concat_minute)]
    else:
        hr_lst = [f"{i:02d}" for i in range(0, 24)]
        minutes = [f"{i:02d}" for i in range(0, 1, concat_minute)]

    print(f"{hr_lst = }")
    print(f"{minutes = }")

    for jday in jdays:
        month = datetime.strptime(f"{year}_{jday}", '%Y_%j').strftime('%m')
        day = datetime.strptime(f"{year}_{jday}", '%Y_%j').strftime('%d')

        for hr in hr_lst:
            for minute in minutes:
                print(f"Processing {year} {month} {day} {hr} {minute}")
                if local_test_mode:
                    cmd = f"python ../../../CCTorch/run.py --pair_list=pair_list/pair_list_500_CC_0_1249.txt --data_list1=data_list/data_list_mb_{year}{month}{day}T{hr}{minute}.txt --sampling_rate=50 --maxlag 120 --mode=AN  --block_size1 1 --block_size2 2000 --domain stft --device=cpu"
                else:
                    path = f"{bucket}/das_preprocess/"
                    print(f"Reading preprocess file list from {path}...")

                    # cmd = f"python /opt/CCTorch/run.py --pair_list=pair_list_dvv_redo.txt --data_list1={path}data_list_3/data_list_mb_{year}{month}{day}T{hr}{minute}.txt --sampling_rate=50 --maxlag 120 --mode=AN  --block_size1 1 --block_size2 2000 --domain stft --device=cpu"
                    cmd = f"python /opt/CCTorch/run.py --pair_list=pair_list_dvv_redo.txt --data_list1={path}data_list_3/data_list_mb_{year}{month}{day}T{hr}{minute}.txt --sampling_rate=50 --maxlag 120 --mode=AN  --block_size1 1 --block_size2 200 --batch_size 128 --domain stft --device=cpu"
                    # cmd = f"python /opt/CCTorch/run.py --pair_list=pair_list/{pair_lst} --data_list1=data_list/data_list_mb_{year}{month}{day}T{hr}{minute}.txt --sampling_rate=50 --maxlag 120 --mode=AN  --block_size1 1 --block_size2 2000 --domain stft --device=cpu"
                cmd += f" --result_path={result_path} --result_file={year}/{year}.{jday}.{hr}.{minute}.zarr"
                print(cmd)
                os.system(cmd)


# %%
