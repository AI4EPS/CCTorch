import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--jday", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--protocol", type=str, default="gs")
    parser.add_argument("--token_file", type=str, default="application_default_credentials.json")
    parser.add_argument("--bucket", type=str, default="gs://cctorch/ambient_noise")
    parser.add_argument("--root_path", type=str, default="./")
    parser.add_argument("--knn_dist", type=int, default=500)
    args = parser.parse_args()

    return args
