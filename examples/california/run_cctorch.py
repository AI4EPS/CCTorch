# %%
import os

import fsspec
import pandas as pd
from args import parse_args

args = parse_args()


# %%
if __name__ == "__main__":
    year = args.year
    node_rank = args.node_rank
    num_nodes = args.num_nodes
    protocol = args.protocol
    token_file = args.token_file
    result_path = args.result_path

    # get how many days in the year
    jdays = pd.date_range(start=f"{year}-11-18", end=f"{year}-11-27").strftime("%j").tolist()

    jdays = jdays[node_rank::num_nodes]
    print(f"{jdays = }")

    fs = fsspec.filesystem(protocol, token=token_file)
    print('result_path = ', result_path)

    for jday in jdays:

        # cmd = f"python mseeds1.py --year {year} --jday {jday}"
        # print(cmd)
        # os.system(cmd)

        # cmd = f"python downsample.py --year {year} --jday {jday}"
        # print(cmd)
        # os.system(cmd)

        cmd = f"python mseeds2.py --year {year} --jday {jday} --token_file {token_file}" 
        print(cmd)
        os.system(cmd)

        with open(f"pairs2_{year}_{jday}.txt", "r") as f:
            if len(f.readlines()) == 0:
                print(f"pairs2_{year}_{jday}.txt is empty")
                continue

        # cmd = f"python /opt/CCTorch/run.py --pair_list=pairs2_{year}_{jday}.txt --data_list1=mseeds2_{year}_{jday}.txt --data_format1=mseed --sampling_rate=20 --mode=AN --maxlag 300  --block_size1 300 --block_size2 300 --batch_size 4  --domain stft --device=cuda"
        cmd = f"python ../../run.py --pair_list=pairs2_{year}_{jday}_MLR.txt --data_list1=mseeds2_{year}_{jday}.txt --data_format1=mseed --sampling_rate=20 --mode=AN --maxlag 300  --block_size1 300 --block_size2 300 --batch_size 4  --domain stft --device=cpu"
        # cmd += f" --result_path={result_path} --result_file={year}/{year}.{jday}.h5"
        cmd += f" --result_path={result_path} --result_file={year}/{year}.{jday}.zarr"
        # cmd += f" --result_path={result_path}/{year}/{year}.{jday}"
        # cmd = f"python ../../run.py --pair_list=pairs2_{year}_{jday}.txt --data_list1=mseeds2_{year}_{jday}.txt --data_format1=mseed --sampling_rate=20 --mode=AN  --block_size1 10 --block_size2 10 --batch_size 1  --domain stft --device=cpu"
        print(cmd)
        os.system(cmd)


# %%
