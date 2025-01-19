# %%
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def get_args_parser(add_help=True):

    import argparse

    parser = argparse.ArgumentParser(description="Read CCTorch Results", add_help=add_help)
    parser.add_argument("--result_path", type=str, default="results", help="path to results")
    parser.add_argument("--figure_path", type=str, default="figures", help="path to figures")
    return parser


# %%
if __name__ == "__main__":

    args = get_args_parser().parse_args()

    result_path = Path(args.result_path)
    figure_path = Path(args.figure_path)
    if not figure_path.exists():
        figure_path.mkdir(parents=True)

    h5_files = sorted(result_path.glob("*.h5"))
    print(f"{len(h5_files)} hdf5 files found")

    data = []
    index = []
    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as fp:
            print(fp.keys())
            ch1_list = fp.keys()
            for ch1 in ch1_list:
                ch2_list = fp[ch1].keys()
                for ch2 in ch2_list:
                    plt.figure()
                    plt.plot(fp[f"{ch1}/{ch2}"]["xcorr"][0, :])
                    plt.plot(fp[f"{ch1}/{ch2}"]["xcorr"][1, :] + 1)
                    plt.plot(fp[f"{ch1}/{ch2}"]["xcorr"][2, :] + 2)
                    plt.savefig(figure_path / f"ambient_noise_{ch1}_{ch2}.png", dpi=300, bbox_inches="tight")
                # raise
                # for ch2 in ch2_list:
                #     data.append(fp[ch1][ch2]["xcorr"][:])
                #     index.append(ch2)

            raise
