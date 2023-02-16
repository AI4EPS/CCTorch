import logging
import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing import Manager
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import functools
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

import utils
from cctorch import (
    CCDataset,
    CCIterableDataset,
    CCModel,)
from cctorch.transforms import *
from cctorch.utils import write_results


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Correlation using Pytorch", add_help=add_help)
    parser.add_argument("--pair-list", default=None, type=str, help="pair list")
    parser.add_argument("--data-list1", default=None, type=str, help="data list 1")
    parser.add_argument("--data-list2", default=None, type=str, help="data list 1")
    parser.add_argument("--data-path", default="./", type=str, help="data path")
    parser.add_argument("--result-path", default="./results", type=str, help="results path")
    parser.add_argument("--dataset-type", default="iterable", type=str, help="data loader type in {map, iterable}")
    parser.add_argument("--block-num1", default=1, type=int, help="Number of blocks for the 1st data pair dimension")
    parser.add_argument("--block-num2", default=1, type=int, help="Number of blocks for the 2nd data pair dimension")
    parser.add_argument("--auto-xcorr", action="store_true", help="do auto-correlation for data list")

    ## ambient noise parameters
    parser.add_argument("--min-channel", default=0, type=int, help="minimum channel index")
    parser.add_argument("--max-channel", default=None, type=int, help="maximum channel index")
    parser.add_argument("--delta-channel", default=1, type=int, help="channel interval")
    parser.add_argument("--left-end-channel", default=None, type=int, help="channel index of the left end from the source")
    parser.add_argument("--right-end-channel", default=None, type=int, help="channel index of the right end from the source")
    parser.add_argument(
        "--fixed-channels",
        nargs="+",
        default=None,
        type=int,
        help="fixed channel index, if specified, min and max are ignored",
    )

    # xcor parameters
    parser.add_argument("--domain", default="time", type=str, help="time domain or frequency domain")
    parser.add_argument("--dt", default=0.01, type=float, help="time sampling interval")
    parser.add_argument("--maxlag", default=0.5, type=float, help="maximum time lag during cross-correlation")
    parser.add_argument("--taper", action="store_true", help="taper two data window")
    parser.add_argument("--interp", action="store_true", help="interpolate the data window along time axs")
    parser.add_argument("--scale-factor", default=10, type=int, help="interpolation scale up factor")
    parser.add_argument(
        "--channel-shift", default=0, type=int, help="channel shift of 2nd window for cross-correlation"
    )
    parser.add_argument("--reduce-t", action="store_true", help="reduce the time axis of xcor data")
    parser.add_argument(
        "--reduce-x",
        action="store_true",
        help="reduce the channel axis of xcor data: only have effect when reduce_t is true",
    )
    parser.add_argument(
        "--mccc", action="store_true", help="use mccc to reduce time axis: only have effect when reduce_t is true"
    )
    parser.add_argument("--phase-type1", default="P", type=str, help="Phase type of the 1st data window")
    parser.add_argument("--phase-type2", default="S", type=str, help="Phase type of the 2nd data window")
    parser.add_argument(
        "--path-xcor-data", default="", type=str, help="path to save xcor data output: path_{channel_shift}"
    )
    parser.add_argument(
        "--path-xcor-pick", default="", type=str, help="path to save xcor pick output: path_{channel_shift}"
    )
    parser.add_argument(
        "--path-xcor-matrix", default="", type=str, help="path to save xcor matrix output: path_{channel_shift}"
    )
    parser.add_argument("--path-dasinfo", default="", type=str, help="csv file with das channel info")

    parser.add_argument(
        "--mode",
        default="CC",
        type=str,
        help="mode for tasks of CC (cross-correlation), TM (template matching), and AM (ambient noise)",
    )
    parser.add_argument("--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("--buffer-size", default=10, type=int, help="buffer size for writing to h5 file")
    parser.add_argument("--workers", default=0, type=int, help="data loading workers")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu, Default: cuda)")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    parser.add_argument("--log-interval", default=10, type=int, help="log every n iterations")
    return parser


def main(args):

    logging.basicConfig(filename="cctorch.log", level=logging.INFO)
    utils.init_distributed_mode(args)
    rank = utils.get_rank() if args.distributed else 0
    world_size = utils.get_world_size() if args.distributed else 1
    device = torch.device(args.device)

    @dataclass
    class CCConfig:
        ### dataset
        mode = args.mode
        auto_xcorr = args.auto_xcorr

        ### ccmodel
        dt = args.dt
        fs = int(1. / dt)
        maxlag = args.maxlag
        nlag = int(maxlag / dt)
        nma = (20, 0)
        reduce_t = args.reduce_t
        reduce_x = args.reduce_x
        channel_shift = args.channel_shift
        mccc = args.mccc
        use_pair_index = True if args.dataset_type == "map" else False
        domain = args.domain

        ### ambinet noise
        max_channel = args.max_channel
        min_channel = args.min_channel
        delta_channel = args.delta_channel
        left_end_channel = args.left_end_channel
        right_end_channel = args.right_end_channel
        fixed_channels = args.fixed_channels
        #### preprocessing for ambient noise
        transforms_on_file = True
        transforms_on_batch = False
        window_time = 40
        window_size = window_time*fs

    ccconfig = CCConfig()

    utils.mkdir(args.result_path)

    preprocess = []
    if args.mode == "CC":
        if args.taper:
            preprocess.append(T.Lambda(taper_time))
        if args.interp:
            preprocess.append(T.Lambda(interp_time_cubic_spline))
        if args.domain == "time":
            preprocess.append(T.Lambda(normalize))
        elif args.domain == "frequency":
            preprocess.append(T.Lambda(fft_real_normalize))
    elif args.mode == "TM":
        ## TODO add preprocess for template matching
        pass
    elif args.mode == "AM":
        ## TODO add preprocess for ambient noise
        # preprocess.append(T.Lambda(remove_median))
        preprocess.append(T.Lambda(functools.partial(moving_normalization, window_size=ccconfig.window_size)))
    preprocess = T.Compose(preprocess)

    postprocess = []
    if args.mode == "CC":
        ## TODO add preprocess for cross-correlation
        pass
    elif args.mode == "TM":
        ## TODO add preprocess for template matching
        pass
    elif args.mode == "AM":
        ## TODO add preprocess for ambient noise
        pass
    postprocess = T.Compose(postprocess)

    if args.dataset_type == "map":
        dataset = CCDataset(
            config=ccconfig,
            pair_list=args.pair_list,
            data_list1=args.data_list1,
            data_list2=args.data_list2,
            block_num1=args.block_num1,
            block_num2=args.block_num2,
            data_path=args.data_path,
            device="cpu" if args.workers > 0 else args.device,
            transforms=preprocess,
            rank=rank,
            world_size=world_size,
        )
    elif args.dataset_type == "iterable": ## prefered
        dataset = CCIterableDataset(
            config=ccconfig,
            pair_list=args.pair_list,
            data_list1=args.data_list1,
            data_list2=args.data_list2,
            block_num1=args.block_num1,
            block_num2=args.block_num2,
            data_path=args.data_path,
            device=args.device,
            transforms=preprocess,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
        )
    else:
        raise ValueError(f"dataset_type {args.dataset_type} not supported")
    # if len(dataset) < world_size:
    #     raise ValueError(f"dataset size {len(dataset)} is smaller than world size {world_size}")

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.workers if args.dataset_type == "map" else 0,
        sampler=sampler if args.dataset_type == "map" else None,
        pin_memory=False,
    )

    ccmodel = CCModel(
        config=ccconfig,
        batch_size=args.batch_size,  ## only useful for dataset_type == map
        to_device=False,  ## to_device is done in dataset in default
        device=args.device,
        transforms=postprocess,
    )
    ccmodel.to(args.device)

    results = []
    num = 0
    metric_logger = utils.MetricLogger(delimiter="  ")
    for data in metric_logger.log_every(dataloader, args.log_interval, ""):
        result = ccmodel(data)
        results.append(result)
        num += 1
        if num % args.buffer_size == 0:
            write_results(results, args.result_path, ccconfig, rank=rank, world_size=world_size)
            results = []
    if num > 0:
        write_results(results, args.result_path, ccconfig, rank=rank, world_size=world_size)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    args = get_args_parser().parse_args()
    main(args)
