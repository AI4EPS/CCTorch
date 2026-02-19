# %%
import os
from args import parse_args
import pandas as pd
import fsspec
import os

args = parse_args()


# %%
if __name__ == "__main__":
    year = args.year
    jday_start = args.jday_start
    jday_end = args.jday_end
    minute_end = args.minute_end
    concat_minute = args.concat_minute
    target_freq = args.target_freq
    node_rank = args.node_rank
    num_nodes = args.num_nodes
    protocol = args.protocol
    token_file = args.token_file
    token_file_load = args.token_file_load

    down_sample_folder = args.downsample_folder
    downsample_type = args.downsample_type

    print("Downsample type:", downsample_type)
    # get how many days in the year
    jday_lst = [f"{i:03d}" for i in range(jday_start, jday_end + 1)]
    jday_lst = jday_lst[node_rank::num_nodes]
    print(f"{jday_lst = }")

    hr_lst = [f"{i:02d}" for i in range(0, 24)]
    print(f"{hr_lst = }")
    minutes = [f"{i:02d}" for i in range(0, minute_end, concat_minute)]
    print(f"{minutes = }")

    fs = fsspec.filesystem(protocol, token=token_file)
    for jday in jday_lst:
        for hr in hr_lst:
            for minute in minutes:
                cmd = f"python downsample_das.py --year {year} --jday {jday} --hr {hr} --minute {minute} --target_freq {target_freq} --token_file_load {token_file_load} --downsample_folder {down_sample_folder} --downsample_type {downsample_type}"
                print(cmd)
                os.system(cmd)
# %%
