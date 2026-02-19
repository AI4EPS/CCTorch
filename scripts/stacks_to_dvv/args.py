import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--file_mark", type=str, default="A")
    parser.add_argument("--bucket_das", type=str, default="gs://cctorch/ambient_noise_das")
    parser.add_argument("--project", type=str, default="california_seasonal_dvv")

    args = parser.parse_args()

    return args
