from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset


class CCDataset(Dataset):
    def __init__(self, pair_list, data_path, shared_dict, device="cpu", transform=None, rank=0, world_size=1, **kwargs):
        super(CCDataset).__init__()
        self.cc_list = pd.read_csv(pair_list, header=None, names=["event1", "event2"])  # .iloc[rank::world_size]
        self.data_path = Path(data_path)
        self.shared_dict = shared_dict
        self.local_dict = {}
        self.transform = transform
        self.device = device

    def _read_das(self, event):
        # if event not in self.shared_dict:
        #     print("Adding {} to shared_dict".format(event))
        if event not in self.local_dict:
            print("Adding {} to local_dict".format(event))
            with h5py.File(self.data_path / f"{event}.h5", "r") as fid:
                data = fid["data"]["P"]["data"][:]
                data = torch.from_numpy(data)

            if self.transform is not None:
                # self.shared_dict[event] = self.transform(data)
                self.local_dict[event] = self.transform(data.cuda())

                ## TODO: check if GPU works and if it is faster
                # if self.device == "cpu":
                #     self.shared_dict[event] = self.transform(data)
                # elif self.device == "cuda":
                #     num_gpu = torch.cuda.device_count()
                #     worker_id = torch.utils.data.get_worker_info().id
                #     self.shared_dict[event] = self.transform(data.cuda(worker_id % num_gpu)).cpu()
                # else:
                #     raise

        # return self.shared_dict[event]
        return self.local_dict[event]

    def __getitem__(self, index):
        event1, event2 = self.cc_list.iloc[index]
        data1 = self._read_das(event1)
        data2 = self._read_das(event2)
        return {"event": event1, "data": data1}, {"event": event2, "data": data2}

    def __len__(self):
        return len(self.cc_list)


class CCIterableDataset(IterableDataset):
    def __init__(
        self,
        data_list1,
        data_list2,
        block_size1,
        block_size2,
        data_path,
        shared_dict,
        device="cpu",
        transform=None,
        rank=0,
        world_size=1,
        **kwargs,
    ):
        super(CCIterableDataset).__init__()
        self.data_list1 = data_list1
        self.data_list2 = data_list2
        self.group1 = [list(x) for x in np.array_split(data_list1, len(data_list1) // block_size1)]
        self.group2 = [list(x) for x in np.array_split(data_list2, len(data_list2) // block_size2)]
        self.block_size1 = block_size1
        self.block_size2 = block_size2
        if len(self.group1) > len(self.group2):
            self.group1, self.group2 = self.group2, self.group1
            self.block_size1, self.block_size2 = self.block_size2, self.block_size1
        self.block_index = [(i, j) for i in range(len(self.group1)) for j in range(i, len(self.group2))][
            rank::world_size
        ]
        self.data_path = Path(data_path)
        self.shared_dict = shared_dict
        self.transform = transform
        self.device = device

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        # return iter(
        #     self.sample(
        #         self.group1[np.array_split(np.arange(len(self.group1)), num_workers)[worker_id]],
        #         self.group2[np.array_split(np.arange(len(self.group2)), num_workers)[worker_id]],
        #     )
        # )
        return iter(self.sample(self.block_index))

    def _read_das(self, event, local_dict):

        if event not in local_dict:
            # print("Adding {} to local_dict".format(event))
            with h5py.File(self.data_path / f"{event}.h5", "r") as fid:
                data = fid["data"]["P"]["data"][:]
                data = torch.from_numpy(data)

            if self.transform is not None:
                local_dict[event] = self.transform(data.cuda())

        return local_dict[event]

    def sample(self, block_index):
        for i, j in block_index:
            local_dict = {}
            if len(self.group1[i]) > len(self.group2[j]):
                event1, event2 = self.group2[i], self.group1[j]
            else:
                event1, event2 = self.group1[i], self.group2[j]
            # print(f"{len(event1) = }, {len(event2) = }")
            for ii in range(len(event1)):
                for jj in range(ii + 1, len(event2)):
                    # print(f"{ii = }, {jj = }")
                    data1 = self._read_das(event1[ii], local_dict)
                    data2 = self._read_das(event2[jj], local_dict)
                    yield {"event": event1[ii], "data": data1}, {"event": event2[jj], "data": data2}

            # keys = list(local_dict.keys())
            # for k in keys:
            #     del local_dict[k]
            del local_dict
            torch.cuda.empty_cache()

    def __len__(self):
        short_list = min(len(self.data_list1), len(self.data_list2))
        return len(self.data_list1) * len(self.data_list2) - short_list * (short_list + 1) // 2
