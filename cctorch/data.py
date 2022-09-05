from pathlib import Path

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset


class CCDataset(Dataset):
    def __init__(self, pair_list, data_path, shared_dict, device="cpu", transform=None, rank=0, world_size=1, **kwargs):
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
