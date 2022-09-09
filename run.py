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
    fft_normalize,
    write_xcor_to_csv,
)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Correlation using Pytorch", add_help=add_help)
    parser.add_argument(
        "--pair-list", default="/home/jxli/packages/CCTorch/tests/pair_more.txt", type=str, help="pair list"
    )
    parser.add_argument(
        "--data-path", default="/kuafu/jxli/Data/DASEventData/Ridgecrest_South/temp3", type=str, help="data path"
    )
    parser.add_argument(
        "--mode",
        default="differential_time",
        type=str,
        help="mode for tasks of differential_time, template_matching, and ambient_noise",
    )
    parser.add_argument("--batch-size", default=8, type=int, help="batch size")
    parser.add_argument("--workers", default=16, type=int, help="data loading workers")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--output-dir", default="tests/ridgecrest", type=str, help="path to save outputs")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    ## TODO: Add more arguments for visualization, data processing, etc
    return parser


def main(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    manager = Manager()
    shared_dict = manager.dict()
    transform = T.Compose([T.Lambda(fft_normalize)])
    # transform = get_transform()

    pair_list = args.pair_list
    data_path = args.data_path

    rank = utils.get_rank() if args.distributed else 0
    world_size = utils.get_world_size() if args.distributed else 1

    # dataset = CCDataset(
    #     pair_list, data_path, shared_dict, device=args.device, transform=transform, rank=rank, world_size=world_size
    # )

    pair_list = pd.read_csv(pair_list, header=None, names=["event1", "event2"])
    data_list1 = list(set(pair_list["event1"].tolist()))
    data_list2 = data_list1
    block_size1 = len(data_list1) // 3
    block_size2 = len(data_list2) // 3
    dataset = CCIterableDataset(
        data_list1=data_list1,
        data_list2=data_list2,
        block_size1=block_size1,
        block_size2=block_size2,
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
    ccmodel = CCModel(device=args.device, to_device=False, batching=None, dt=0.01, maxlag=0.3)
    ccmodel.to(device)
    # if args.distributed:
    #     # ccmodel = torch.nn.parallel.DistributedDataParallel(ccmodel, device_ids=[args.gpu])
    #     # model_without_ddp = ccmodel.module
    #     pass
    # else:
    #     ccmodel = nn.DataParallel(ccmodel)

    metric_logger = utils.MetricLogger(delimiter="  ")
    for x in metric_logger.log_every(dataloader, 100, "CC: "):
        # print(x[0]["data"].shape)
        # print(x[1]["data"].shape)
        result = ccmodel(x)
        # write_xcor_to_csv(result, args.output_dir)
        ## TODO: ADD post-processing
        ## TODO: Add visualization


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    args = get_args_parser().parse_args()
    main(args)
