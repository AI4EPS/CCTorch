# %%
import os
from args import parse_args
import pandas as pd

args = parse_args()

# %%
if __name__ == "__main__":
    year = args.year
    node_rank = args.node_rank
    num_nodes = args.num_nodes
    protocol = args.protocol
    token_file = args.token_file

    # get how many days in the year
    jdays = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31").strftime("%j").tolist()

    jdays = jdays[node_rank::num_nodes]
    print(f"{jdays = }")

    for jday in jdays:
        cmd = f"python downsample.py --year {year} --jday {jday}"
        print(cmd)
        os.system(cmd)

        cmd = f"python mseeds2.py --year {year} --jday {jday}"
        print(cmd)
        os.system(cmd)

        cmd = f"python /opt/CCTorch/run.py --pair_list=pairs2_{year}_{jday}.txt --data_list1=mseeds2_{year}_{jday}.txt --data_format1=mseed --sampling_rate=20 --mode=AN  --block_size1 300 --block_size2 300 --batch_size 1  --domain stft --device=cuda"
        # cmd = f"python ../../run.py --pair_list=pairs2_{year}_{jday}.txt --data_list1=mseeds2_{year}_{jday}.txt --data_format1=mseed --sampling_rate=20 --mode=AN  --block_size1 10 --block_size2 10 --batch_size 1  --domain stft --device=cpu"
        print(cmd)
        os.system(cmd)


# %%
