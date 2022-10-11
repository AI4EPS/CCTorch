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
    parser.add_argument(
        "--pair-list",
        default="/home/jxli/packages/CCTorch/tests/pair_mammoth_ccfm_test.txt",
        type=str,
        help="pair list",
    )
    parser.add_argument(
        "--path-data", default="/kuafu/jxli/Data/DASEventData/mammoth_south/temp", type=str, help="data path"
    )

    parser.add_argument("--block_num1", default=3, type=int, help="Number of blocks for the 1st data pair dimension")
    parser.add_argument("--block_num2", default=3, type=int, help="Number of blocks for the 2nd data pair dimension")
    parser.add_argument(
        "--generate-pair",
        action="store_true",
        help="generate full pair list from data_list1 and data_list2 if turning on this option",
    )
    parser.add_argument("--auto-xcor", action="store_true", help="do auto-correlation for data list")

    # xcor parameters
    parser.add_argument("--domain", default="time", type=str, help="time domain or frequency domain")
    parser.add_argument("--maxlag", default=0.5, type=float, help="maximum time lag during cross-correlation")
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
        default="differential_time",
        type=str,
        help="mode for tasks of differential_time, template_matching, and ambient_noise",
    )
    parser.add_argument("--batch-size", default=8, type=int, help="batch size")
    parser.add_argument("--workers", default=16, type=int, help="data loading workers")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    ## TODO: Add more arguments for visualization, data processing, etc
    return parser


def main(args):

    if args.path_xcor_data:
        path_xcor_data = f"{args.path_xcor_data}_{args.channel_shift}"
        utils.mkdir(path_xcor_data)
    if args.path_xcor_pick:
        path_xcor_pick = f"{args.path_xcor_pick}_{args.channel_shift}"
        utils.mkdir(path_xcor_pick)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    manager = Manager()
    shared_dict = manager.dict()

    if args.domain == "time":
        transform = T.Compose([T.Lambda(taper_time), T.Lambda(interp_time_cubic_spline), T.Lambda(normalize)])
    elif args.domain == "frequency":
        transform = T.Compose([T.Lambda(taper_time), T.Lambda(interp_time_cubic_spline), T.Lambda(fft_real_normalize)])
    # transform = get_transform()

    pair_list = args.pair_list
    data_path = args.path_data

    rank = utils.get_rank() if args.distributed else 0
    world_size = utils.get_world_size() if args.distributed else 1

    # dataset = CCDataset(
    #     pair_list, data_path, shared_dict, device=args.device, transform=transform, rank=rank, world_size=world_size
    # )

    dataset = CCIterableDataset(
        pair_list=pair_list,
        generate_pair=args.generate_pair,
        auto_xcor=args.auto_xcor,
        block_num1=args.block_num1,
        block_num2=args.block_num2,
        data_path=data_path,
        shared_dict=shared_dict,
        device=args.device,
        transform=transform,
        rank=rank,
        world_size=world_size,
    )
    # print(f"{len(dataset) = }")

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        # batch_size=args.batch_size,
        # num_workers=args.workers,
        batch_size=None,
        num_workers=0,
        # sampler=sampler,
        sampler=None,
        # pin_memory=True
        pin_memory=False,
    )

    ## TODO: check if DataParallel is better for dataset memory
    ccmodel = CCModel(
        device=args.device,
        to_device=False,
        batching=None,
        dt=0.001,
        maxlag=args.maxlag,
        reduce_t=args.reduce_t,
        reduce_x=args.reduce_x,
        channel_shift=args.channel_shift,
        mccc=args.mccc,
        domain=args.domain,
    )
    ccmodel.to(device)
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

    for x in metric_logger.log_every(dataloader, 100, "CC: "):
        # print(x[0]["data"].shape)
        # print(x[1]["data"].shape)
        result = ccmodel(x)
        # write xcor to file
        if args.path_xcor_data:
            write_xcor_data_to_h5(result, path_xcor_data, phase1=args.phase_type1, phase2=args.phase_type1)
        if args.path_xcor_pick and args.mccc:
            write_xcor_mccc_pick_to_csv(result, x, path_xcor_pick, channel_index=channel_index)
        if args.path_xcor_matrix:
            write_xcor_to_ccmat(result, cc_matrix, id_row, id_col)
        ## TODO: ADD post-processing
        ## TODO: Add visualization

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
