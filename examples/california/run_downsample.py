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
    node_rank = args.node_rank
    num_nodes = args.num_nodes
    protocol = args.protocol
    token_file = args.token_file

    # get how many days in the year
    jdays = pd.date_range(start=f"{year}-05-01", end=f"{year}-12-31").strftime("%j").tolist()

    jdays = jdays[node_rank::num_nodes]
    print(f"{jdays = }")

    fs = fsspec.filesystem(protocol, token=token_file)

    for jday in jdays:

        cmd = f"python mseeds1.py --year {year} --jday {jday} --protocol {protocol} --token_file {token_file} --bucket {args.bucket}"
        print(cmd)
        os.system(cmd)

        cmd = f"python downsample.py --year {year} --jday {jday} --protocol {protocol} --token_file {token_file} --bucket {args.bucket}"
        print(cmd)
        os.system(cmd)

        # cmd = f"python mseeds2.py --year {year} --jday {jday}"
        # print(cmd)
        # os.system(cmd)

        # cmd = f"python /opt/CCTorch/run.py --pair_list=pairs2_{year}_{jday}.txt --data_list1=mseeds2_{year}_{jday}.txt --data_format1=mseed --sampling_rate=20 --mode=AN --maxlag 300 --dt 0.05 --block_size1 300 --block_size2 300 --batch_size 1  --domain stft --device=cuda"
        # # cmd = f"python ../../run.py --pair_list=pairs2_{year}_{jday}.txt --data_list1=mseeds2_{year}_{jday}.txt --data_format1=mseed --sampling_rate=20 --mode=AN  --block_size1 10 --block_size2 10 --batch_size 1  --domain stft --device=cpu"
        # print(cmd)
        # os.system(cmd)

        # fs.put(f"results/{year}.{jday}.h5", f"gs://cctorch/ambient_noise/ccf/{year}/{year}.{jday}.h5")
        # os.system(f"rm -f results/{year}.{jday}.h5")


# %%
