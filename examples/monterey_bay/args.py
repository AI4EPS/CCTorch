import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--jday", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--protocol", type=str, default="gs")
    parser.add_argument("--token_file", type=str, default="application_default_credentials.json")
    parser.add_argument("--bucket", type=str, default="gs://cctorch/ambient_noise")
    parser.add_argument("--root_path", type=str, default="./")
    parser.add_argument("--result_path", type=str, default="./results")
    parser.add_argument("--knn_dist", type=int, default=300)

    # --- add for more flexibility ---
    parser.add_argument("--year_start", type=int, default=2025)
    parser.add_argument("--year_end", type=int, default=2025)
    parser.add_argument("--jday_start", type=int, default=1)
    parser.add_argument("--jday_end", type=int, default=1)
    parser.add_argument("--hr", type=int, default=0)
    parser.add_argument("--hr_end", type=int, default=24)
    parser.add_argument("--minute", type=int, default=0)
    parser.add_argument("--minute_end", type=int, default=60)
    parser.add_argument("--concat_minute", type=int, default=4)
    parser.add_argument("--raw_freq", type=float, default=200)
    parser.add_argument("--target_freq", type=float, default=50)
    parser.add_argument("--protocol_load", type=str, default="gcs")
    parser.add_argument("--protocol_save", type=str, default="gs")
    parser.add_argument("--token_file_load", type=str, default="google_default")
    parser.add_argument("--token_file_save", type=str, default="google_default")
    parser.add_argument("--downsample_folder", type=str, default="mbdas_h5_downsample_list")
    parser.add_argument("--downsample_type", type=str, default="raw")
    parser.add_argument("--pair_lst", type=str, default="pair_list.txt")
    parser.add_argument("--file_mark", type=str, default="")
    parser.add_argument("--bucket_das", type=str, default="gs://cctorch/ambient_noise_das")
    parser.add_argument("--project", type=str, default="california_seasonal_dvv")

    args = parser.parse_args()

    return args
