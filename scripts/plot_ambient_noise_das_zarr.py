# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from tqdm.auto import tqdm


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Read CCTorch Results", add_help=add_help)
    parser.add_argument("--result_path", type=str, default="results", help="path to results")
    parser.add_argument("--result_file", type=str, default="test.zarr", help="specific result file to process")
    parser.add_argument("--figure_path", type=str, default="figures", help="path to figures")
    parser.add_argument(
        "--fixed_channels",
        nargs="+",
        default=None,
        type=int,
        help="fixed channel index, if specified, min and max are ignored",
    )
    return parser


# %%
if __name__ == "__main__":
    args = get_args_parser().parse_args()

    result_path = Path(args.result_path)
    figure_path = Path(args.figure_path)
    if not figure_path.exists():
        figure_path.mkdir(parents=True)

    # Find Zarr stores (directories with .zarr or .h5 extension that contain Zarr data)
    zarr_files = sorted(result_path.glob("*.zarr"))
    if len(zarr_files) == 0:
        # Also check for .h5 extensions that might be Zarr stores
        zarr_files = sorted(result_path.glob("*.h5"))
    print(f"{len(zarr_files)} zarr files found: {[f.name for f in zarr_files]}")

    tmp = []
    for ch1 in args.fixed_channels:
        data = []
        index = []
        for zarr_file in tqdm(zarr_files, desc=f"Processing channel {ch1}"):
            # Open Zarr store
            store = zarr.storage.LocalStore(str(zarr_file))
            root = zarr.open_group(store=store, mode="r")

            # Get all channel pairs where ch1 is the first channel
            ch1_str = str(ch1)
            if ch1_str not in root:
                continue

            ch2_list = sorted([int(x) for x in root[ch1_str].keys()])
            for ch2 in ch2_list:
                # Read the cross-correlation data
                xcorr_data = root[ch1_str][str(ch2)][:]
                data.append(xcorr_data)
                index.append(ch2)

        if len(data) == 0:
            print(f"No data found for channel {ch1}")
            continue

        index = np.array(index)
        data = np.stack(data)
        sorted_idx = np.argsort(index)
        index = index[sorted_idx]
        data = data[sorted_idx]

        np.savez(figure_path / f"ambient_noise_das_{ch1}.npz", data=data, index=index)
        plt.figure()
        vmax = np.std(data)
        plt.imshow(data, vmin=-vmax, vmax=vmax, aspect="auto", cmap="RdBu")
        plt.colorbar()
        plt.savefig(figure_path / f"ambient_noise_das_{ch1}.png", dpi=300, bbox_inches="tight")
