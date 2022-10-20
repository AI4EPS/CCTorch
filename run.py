import logging
import multiprocessing as mp
from multiprocessing import Manager
from pathlib import Path

import h5py
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

import utils
from cctorch import (
    CCDataset,
    CCIterableDataset,
    CCModel,
    data,
    fft_real_normalize,
    interp_time_cubic_spline,
    normalize,
    reduce_ccmat,
    taper_time,
    write_xcor_data_to_h5,
    write_xcor_mccc_pick_to_csv,
    write_xcor_to_ccmat,
)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Correlation using Pytorch", add_help=add_help)
    parser.add_argument("--pair-list", default=None, type=str, help="pair list")
    parser.add_argument("--data-list1", default=None, type=str, help="data list 1")
    parser.add_argument("--data-list2", default=None, type=str, help="data list 1")
    parser.add_argument("--data-path", default="./", type=str, help="data path")
    parser.add_argument("--dataset-type", default="map", type=str, help="data loader type in {map, iterable}")
    parser.add_argument("--block_num1", default=1, type=int, help="Number of blocks for the 1st data pair dimension")
    parser.add_argument("--block_num2", default=1, type=int, help="Number of blocks for the 2nd data pair dimension")
    parser.add_argument("--auto-xcorr", action="store_true", help="do auto-correlation for data list")

    # xcor parameters
    parser.add_argument("--domain", default="time", type=str, help="time domain or frequency domain")
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
    parser.add_argument("--phase-type2", default="P", type=str, help="Phase type of the 2nd data window")
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
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")
    parser.add_argument("--workers", default=0, type=int, help="data loading workers")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu, Default: cuda)")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    return parser


def main(args):

    logging.basicConfig(filename="cctorch.log", level=logging.INFO)
    utils.init_distributed_mode(args)
    rank = utils.get_rank() if args.distributed else 0
    world_size = utils.get_world_size() if args.distributed else 1
    device = torch.device(args.device)
    print(args)

    if utils.is_main_process():
        if args.path_xcor_data:
            path_xcor_data = f"{args.path_xcor_data}_{args.channel_shift}"
            utils.mkdir(path_xcor_data)
        if args.path_xcor_pick:
            path_xcor_pick = f"{args.path_xcor_pick}_{args.channel_shift}"
            utils.mkdir(path_xcor_pick)

    transform_list = []
    if args.taper:
        transform_list.append(T.Lambda(taper_time))
    if args.interp:
        transform_list.append(T.Lambda(interp_time_cubic_spline))
    if args.domain == "time":
        transform_list.append(T.Lambda(normalize))
    elif args.domain == "frequency":
        transform_list.append(T.Lambda(fft_real_normalize))
    transform = T.Compose(transform_list)

    if args.dataset_type == "map":
        dataset = CCDataset(
            pair_list=args.pair_list,
            data_list1=args.data_list1,
            data_list2=args.data_list2,
            auto_xcorr=args.auto_xcorr,
            block_num1=args.block_num1,
            block_num2=args.block_num2,
            data_path=args.data_path,
            device=args.device,
            transform=transform,
            rank=rank,
            world_size=world_size,
        )
    elif args.dataset_type == "iterable":
        dataset = CCIterableDataset(
            pair_list=args.pair_list,
            data_list1=args.data_list1,
            data_list2=args.data_list2,
            auto_xcorr=args.auto_xcorr,
            block_num1=args.block_num1,
            block_num2=args.block_num2,
            data_path=args.data_path,
            device=args.device,
            transform=transform,
            rank=rank,
            world_size=world_size,
        )
    else:
        raise ValueError(f"dataset_type {args.dataset_type} not supported")
    print(f"{len(dataset) = }")

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.workers,
        sampler=sampler if args.dataset_type == "Dataset" else None,
        pin_memory=False,
    )

    ccmodel = CCModel(
        dt=0.001,
        maxlag=args.maxlag,
        reduce_t=args.reduce_t,
        reduce_x=args.reduce_x,
        channel_shift=args.channel_shift,
        mccc=args.mccc,
        domain=args.domain,
        use_pair_index=True if args.dataset_type == "map" else False,
        device=args.device,
        batching=False,
    )
    ccmodel.to(args.device)

    ## only used for model with parameters
    # if args.distributed:
    #     # ccmodel = torch.nn.parallel.DistributedDataParallel(ccmodel, device_ids=[args.gpu])
    #     # model_without_ddp = ccmodel.module
    #     pass
    # else:
    #     ccmodel = nn.DataParallel(ccmodel)

    metric_logger = utils.MetricLogger(delimiter="  ")

    if args.path_xcor_matrix:
        cc_matrix = torch.zeros([len(dataset.data_list1), len(dataset.data_list2)], device=args.device)
        id_row = torch.tensor(dataset.data_list1, device=args.device)
        id_col = torch.tensor(dataset.data_list2, device=args.device)

    if args.path_dasinfo:
        channel_index = pd.read_csv(args.path_dasinfo)["index"]
    else:
        channel_index = None

    writing_processes = []
    ctx = mp.get_context("spawn")
    ncpu = mp.cpu_count()
    for x in metric_logger.log_every(dataloader, 1, "CC: "):
        result = ccmodel(x)

        if args.path_xcor_data:
            # write_xcor_data_to_h5(result, path_xcor_data, phase1=args.phase_type1, phase2=args.phase_type1)
            for k in result:
                result[k] = result[k].cpu()
            p = ctx.Process(
                target=write_xcor_data_to_h5,
                args=(
                    result,
                    path_xcor_data,
                ),
                kwargs={"phase1": args.phase_type1, "phase2": args.phase_type1},
            )
            p.start()
            writing_processes.append(p)
        if args.path_xcor_pick and args.mccc:
            # write_xcor_mccc_pick_to_csv(result, x, path_xcor_pick, channel_index=channel_index)
            p = ctx.Process(target=write_xcor_mccc_pick_to_csv, args=(result, x, path_xcor_pick, channel_index))
            p.start()
            writing_processes.append(p)
        if args.path_xcor_matrix:
            # write_xcor_to_ccmat(result, cc_matrix, id_row, id_col)
            p = ctx.Process(target=write_xcor_mccc_pick_to_csv, args=(result, x, path_xcor_pick, channel_index))
            p.start()
            writing_processes.append(p)

        ## prevent too many processes
        if len(writing_processes) > ncpu:
            for p in writing_processes:
                p.join()
            writing_processes = []

        ## TODO: ADD post-processing
        ## TODO: Add visualization

    logging.info("Waiting for writing processes to finish...")
    for p in writing_processes:
        p.join()
        logging.info("Finish writing outputs.")

    if args.path_xcor_matrix:
        import numpy as np

        cc_matrix = cc_matrix.cpu().numpy()
        np.savez(
            f"{args.path_xcor_matrix}_{args.channel_shift}_{rank}.npz",
            cc=cc_matrix,
            id_row=dataset.data_list1,
            id_col=dataset.data_list2,
            id_pair=list(dataset.pairs),
        )

        torch.distributed.barrier()
        if rank == 0:
            reduce_ccmat(args.path_xcor_matrix, args.channel_shift, world_size)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    args = get_args_parser().parse_args()
    main(args)
