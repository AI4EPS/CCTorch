from multiprocessing import Manager
from pathlib import Path
from cctorch.postprocess import write_xcor_to_ccmat, write_xcor_to_h5, write_xcor_mccc_pick_to_csv
from cctorch.transforms import interp_time_cubic_spline, pick_mccc_refine

import h5py
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datetime import datetime, timedelta

import utils
from cctorch import (
    CCDataset,
    CCIterableDataset,
    CCModel,
    data,
    fft_real_normalize,
    write_xcor_to_csv,
)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Correlation using Pytorch", add_help=add_help)
    parser.add_argument(
        "--pair-list", default="/home/jxli/packages/CCTorch/tests/pair_mammoth_xcor.txt", type=str, help="pair list"
    )
    parser.add_argument(
        "--data-path", default="/kuafu/jxli/Data/DASEventData/mammoth_south/temp", type=str, help="data path"
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
    parser.add_argument("--output-dir", default="/export/ssd-tmp-nobak2/jxli/DASEventData/mammoth_south/temp_xcor/", type=str, help="path to save outputs")

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
    transform = T.Compose([T.Lambda(fft_real_normalize)])
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
    ccmodel = CCModel(device=args.device, to_device=False, batching=None, dt=0.01, maxlag=0.5, reduce_t=True, channel_shift=0, mccc=True)
    ccmodel.to(device)
    # if args.distributed:
    #     # ccmodel = torch.nn.parallel.DistributedDataParallel(ccmodel, device_ids=[args.gpu])
    #     # model_without_ddp = ccmodel.module
    #     pass
    # else:
    #     ccmodel = nn.DataParallel(ccmodel)

    metric_logger = utils.MetricLogger(delimiter="  ")
    #cc_matrix = torch.zeros([len(data_list1), len(data_list2)]).cuda()
    #id_row = torch.tensor(data_list1).cuda()
    #id_col = torch.tensor(data_list2).cuda()
    channel_index = pd.read_csv("/kuafu/EventData/Mammoth_south/das_info.csv")['index']

    path_output_pick = f'{args.output_dir}/picks_{0}'
    path_output_xcor = f'{args.output_dir}/xcorr_{0}'
    for x in metric_logger.log_every(dataloader, 100, "CC: "):
        # print(x[0]["data"].shape)
        # print(x[1]["data"].shape)
        result = ccmodel(x)
        # pick via mccc
        write_xcor_mccc_pick_to_csv(result, x, path_output_pick, channel_index=channel_index)
        write_xcor_to_h5(result, path_output_xcor, phase1='P', phase2='P')
        # write_xcor_to_ccmat(result, cc_matrix, id_row, id_col)
        # write_xcor_to_csv(result, args.output_dir)
        ## TODO: ADD post-processing
        ## TODO: Add visualization

    #cc_matrix = cc_matrix.cpu().numpy()
    #import numpy as np
    #np.savez(f'./tests/mammoth_ccmat_rank_{rank}.npz', cc=cc_matrix, id_row=data_list1, id_col=data_list2)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    args = get_args_parser().parse_args()
    main(args)
