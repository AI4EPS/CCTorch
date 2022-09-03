from multiprocessing import Manager
from pathlib import Path

import h5py
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

import utils
from cctorch import FFT, CCDataset, CCModel


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Cross-correlation using Pytorch", add_help=add_help)
    parser.add_argument("--pair-list", default="tests/pair_ridgecrest.txt", type=str, help="pair list")
    parser.add_argument(
        "--data-path", default="/kuafu/jxli/Data/DASEventData/Ridgecrest_South/temp3", type=str, help="data path"
    )
    parser.add_argument("--batch-size", default=8, type=int, help="batch size")
    parser.add_argument("--workers", default=16, type=int, help="data loading workers")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")

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
    transform = T.Compose([T.Lambda(FFT)])
    # transform = get_transform()

    pair_list = args.pair_list
    data_path = args.data_path
    dataset = CCDataset(pair_list, data_path, shared_dict, device=args.device, transform=transform)

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers, sampler=sampler, pin_memory=True
    )

    ## TODO: check if DataParallel is better for dataset memory
    ccmodel = CCModel(device=args.device, dt=0.01, maxlag=0.3)
    ccmodel.to(device)
    if args.distributed:
        # ccmodel = torch.nn.parallel.DistributedDataParallel(ccmodel, device_ids=[args.gpu])
        # model_without_ddp = ccmodel.module
        pass
    else:
        ccmodel = nn.DataParallel(ccmodel)

    for x in dataloader:
        print(x[0]["data"].shape)
        print(x[1]["data"].shape)
        y = ccmodel(x)
        ## TODO: ADD post-processing
        ## TODO: Add visualization


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    args = get_args_parser().parse_args()
    main(args)
