from multiprocessing import Manager
from pathlib import Path

import h5py
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

import utils


class CCDataset(Dataset):
    def __init__(self, pair_list, data_path, shared_dict, device="cpu", transform=None):
        self.cc_list = pd.read_csv(pair_list, header=None, names=["event1", "event2"])
        self.data_path = Path(data_path)
        self.shared_dict = shared_dict
        self.transform = transform
        self.device = device

    def _read_das(self, event):
        if event not in self.shared_dict:
            print("Adding {} to shared_dict".format(event))
            with h5py.File(self.data_path / event, "r") as fp:
                data = fp["data"][:, :]
                data = torch.from_numpy(data)

            if self.transform is not None:
                ## TODO: check if GPU works and if it is faster
                if self.device == "cpu":
                    self.shared_dict[event] = self.transform(data)
                elif self.device == "cuda":
                    num_gpu = torch.cuda.device_count()
                    worker_id = torch.utils.data.get_worker_info().id
                    self.shared_dict[event] = self.transform(data.cuda(worker_id % num_gpu)).cpu()
                else:
                    raise

        return self.shared_dict[event]

    def __getitem__(self, index):
        event1, event2 = self.cc_list.iloc[index]
        data1 = self._read_das(event1)
        data2 = self._read_das(event2)

        return {"event": event1, "data": data1}, {"event": event2, "data": data2}

    def __len__(self):
        return len(self.cc_list)


def FFT(x):
    ## TODO: check FFT is correct
    return torch.fft.rfft(x, 1)


# def get_transform() -> callable:
#     return torch.nn.Sequential(
#         torch.fft.rfft,
#     )


class CCModel(nn.Module):
    def __init__(self, device):
        super(CCModel, self).__init__()
        self.device = device

    def forward(self, x):
        x1, x2 = x
        data1 = x1["data"].to(self.device)
        data2 = x2["data"].to(self.device)
        print(data1.device, data2.device)
        ## TODO: Implement CC

        ## TODO: Discuss return data format
        # return {"dt": [], "cc": []}


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Cross-correlation using Pytorch", add_help=add_help)
    parser.add_argument("--pair-list", default="tests/pairs.txt", type=str, help="pair list")
    parser.add_argument("--data-path", default="/kuafu/EventData/Mammoth_south/data", type=str, help="data path")
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
    ccmodel = CCModel(device=args.device)
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
