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
    hr = args.hr
    hr_end = args.hr_end
    minute = args.minute
    concat_minute = args.concat_minute
    node_rank = args.node_rank
    num_nodes = args.num_nodes
    protocol = args.protocol
    token_file = args.token_file

    # get how many days in the year
    if (hr + minute) > 0:
        jday_lst = [f"{i:03d}" for i in range(jday_start, jday_end+1)]
    else:
        # jday_lst = [f"{i:03d}" for i in range(jday_start, jday_start+1)]
        jday_lst = [f"{i:03d}" for i in range(jday_start, jday_end+1)]
    jday_lst = jday_lst[node_rank::num_nodes]

    print(f"{jday_lst = }")

    if minute > 0:
        hr_lst = [f"{i:02d}" for i in range(hr, hr+1)]
    else:
        hr_lst = [f"{i:02d}" for i in range(hr, hr_end)]
    print(f"{hr_lst = }")

    minutes = [f"{i:02d}" for i in range(minute-minute%concat_minute, 60, concat_minute)]

    fs = fsspec.filesystem(protocol, token=token_file)
    for jday in jday_lst:
        for hr in hr_lst:
            for minute in minutes:
                cmd = f"python downchannel_das.py --year {year} --jday {jday} --hr {hr} --minute {minute}"
                print(cmd)
                os.system(cmd)
# %%
