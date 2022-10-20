from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset


class CCDataset(Dataset):
    def __init__(
        self,
        pair_list=None,
        data_list1=None,
        data_list2=None,
        data_path="./",
        data_format="h5",
        block_num1=1,
        block_num2=1,
        auto_xcorr=False,
        device="cpu",
        transform=None,
        rank=0,
        world_size=1,
        **kwargs,
    ):
        super(CCDataset).__init__()
        ## TODO: extract this part into a function; keep this temporary until TM and AN are implemented
        ## pair_list has the highest priority
        if pair_list is not None:
            self.pair_list, self.data_list1, self.data_list2 = read_pair_list(pair_list)
        ## use data_list1 if exits and use pair_list as filtering
        if data_list1 is not None:
            self.data_list1 = pd.unique(pd.read_csv(data_list1, header=None)[0]).tolist()
            if data_list2 is not None:
                self.data_list2 = pd.unique(pd.read_csv(data_list2, header=None)[0]).tolist()
            elif pair_list is None:
                self.data_list2 = self.data_list1
            ## generate pair_list if not provided
            if pair_list is None:
                self.pair_list = generate_pairs(self.data_list1, self.data_list2)

        self.auto_xcorr = auto_xcorr
        self.block_num1 = block_num1
        self.block_num2 = block_num2
        self.group1 = [list(x) for x in np.array_split(self.data_list1, block_num1) if len(x) > 0]
        self.group2 = [list(x) for x in np.array_split(self.data_list2, block_num2) if len(x) > 0]
        self.block_index = generate_block_index(self.group1, self.group2, self.pair_list)
        self.data_path = Path(data_path)
        self.data_format = data_format
        self.transform = transform
        self.device = device

    def __getitem__(self, idx):

        i, j = self.block_index[idx]
        event1, event2 = self.group1[i], self.group2[j]

        index_dict = {}
        data, info, pair_index = [], [], []
        for ii in range(len(event1)):
            for jj in range(len(event2)):
                if (event1[ii], event2[jj]) not in self.pair_list:
                    continue

                if event1[ii] not in index_dict:
                    data_dict = read_data(event1[ii], self.data_path, self.data_format)
                    data.append(torch.tensor(data_dict["data"]))
                    info.append(data_dict["info"])
                    index_dict[event1[ii]] = len(data) - 1
                idx1 = index_dict[event1[ii]]

                if event2[jj] not in index_dict:
                    data_dict = read_data(event2[jj], self.data_path, self.data_format)
                    data.append(torch.tensor(data_dict["data"]))
                    info.append(data_dict["info"])
                    index_dict[event2[jj]] = len(data) - 1
                idx2 = index_dict[event2[jj]]

                pair_index.append([idx1, idx2])

        if len(data) > 0:
            data = torch.stack(data, dim=0).to(self.device)
            pair_index = torch.tensor(pair_index).to(self.device)

            if self.transform is not None:
                data = self.transform(data)
        else:
            data = torch.empty((1, 1, 1), dtype=torch.float32).to(self.device)
            pair_index = torch.tensor([[0, 0]], dtype=torch.int64).to(self.device)

        return {"data": data, "info": info, "pair_index": pair_index}

    def __len__(self):
        return len(self.block_index)


class CCIterableDataset(IterableDataset):
    def __init__(
        self,
        pair_list=None,
        data_list1=None,
        data_list2=None,
        data_path="./",
        data_format="h5",
        block_num1=1,
        block_num2=1,
        auto_xcorr=False,
        device="cpu",
        transform=None,
        batch_size=32,
        rank=0,
        world_size=1,
        **kwargs,
    ):
        super(CCIterableDataset).__init__()
        ## TODO: extract this part into a function; keep this temporary until TM and AN are implemented
        ## pair_list has the highest priority
        if pair_list is not None:
            self.pair_list, self.data_list1, self.data_list2 = read_pair_list(pair_list)
        ## use data_list1 if exits and use pair_list as filtering
        if data_list1 is not None:
            self.data_list1 = pd.unique(pd.read_csv(data_list1, header=None)[0]).tolist()
            if data_list2 is not None:
                self.data_list2 = pd.unique(pd.read_csv(data_list2, header=None)[0]).tolist()
            elif pair_list is None:
                self.data_list2 = self.data_list1
            ## generate pair_list if not provided
            if pair_list is None:
                self.pair_list = generate_pairs(self.data_list1, self.data_list2)

        self.auto_xcorr = auto_xcorr
        self.block_num1 = block_num1
        self.block_num2 = block_num2
        self.group1 = [list(x) for x in np.array_split(self.data_list1, block_num1) if len(x) > 0]
        self.group2 = [list(x) for x in np.array_split(self.data_list2, block_num2) if len(x) > 0]
        self.block_index = generate_block_index(self.group1, self.group2, self.pair_list)[rank::world_size]
        self.data_path = Path(data_path)
        self.data_format = data_format
        self.transform = transform
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        return iter(self.sample(self.block_index[worker_id::num_workers]))

    def sample(self, block_index):

        for i, j in block_index:
            local_dict = {}
            event1, event2 = self.group1[i], self.group2[j]
            data1, info1, data2, info2 = [], [], [], []
            num = 0
            for ii in range(len(event1)):
                for jj in range(len(event2)):
                    if (event1[ii], event2[jj]) not in self.pair_list:
                        continue

                    if event1[ii] not in local_dict:
                        data_dict1 = read_data(event1[ii], self.data_path, self.data_format)
                        data_dict1["data"] = torch.tensor(data_dict1["data"]).to(self.device)
                        local_dict[event1[ii]] = data_dict1
                    else:
                        data_dict1 = local_dict[event1[ii]]

                    if event2[jj] not in local_dict:
                        data_dict2 = read_data(event2[jj], self.data_path, self.data_format)
                        data_dict2["data"] = torch.tensor(data_dict2["data"]).to(self.device)
                        local_dict[event2[jj]] = data_dict2
                    else:
                        data_dict2 = local_dict[event2[jj]]

                    data1.append(data_dict1["data"])
                    info1.append(data_dict1["info"])
                    data2.append(data_dict2["data"])
                    info2.append(data_dict2["info"])

                    num += 1
                    if num == self.batch_size:
                        num = 0
                        data_batch1 = torch.stack(data1, dim=0)
                        data_batch2 = torch.stack(data2, dim=0)
                        data1, info1, data2, info2 = [], [], [], []
                        if self.transform is not None:
                            data_batch1 = self.transform(data_batch1)
                            data_batch2 = self.transform(data_batch2)
                        ## TODO: fix yield for batch size
                        yield {
                            "event": event1[ii],
                            "data": data_batch1,
                            "event_time": data_dict1["info"]["event"]["event_time"],
                            "shift_index": data_dict1["info"]["shift_index"],
                        }, {
                            "event": event2[jj],
                            "data": data_batch2,
                            "event_time": data_dict2["info"]["event"]["event_time"],
                            "shift_index": data_dict2["info"]["shift_index"],
                        }

            if num > 0:
                data_batch1 = torch.stack(data1, dim=0).to(self.device)
                data_batch2 = torch.stack(data2, dim=0).to(self.device)
                if self.transform is not None:
                    data_batch1 = self.transform(data_batch1)
                    data_batch2 = self.transform(data_batch2)
                ## TODO: fix yield for batch size
                yield {
                    "event": event1[ii],
                    "data": data_batch1,
                    "event_time": data_dict1["info"]["event"]["event_time"],
                    "shift_index": data_dict1["info"]["shift_index"],
                }, {
                    "event": event2[jj],
                    "data": data_batch2,
                    "event_time": data_dict2["info"]["event"]["event_time"],
                    "shift_index": data_dict2["info"]["shift_index"],
                }

            del local_dict
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def __len__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        num_samples = 0
        for i, j in self.block_index[worker_id::num_workers]:
            event1, event2 = self.group1[i], self.group2[j]
            num = 0
            for ii in range(len(event1)):
                for jj in range(len(event2)):
                    if (event1[ii], event2[jj]) not in self.pair_list:
                        continue
                    num += 1
                    if num == self.batch_size:
                        num = 0
                        num_samples += 1
            if num > 0:
                num_samples += 1

        return num_samples


def generate_pairs(event1, event2, auto_xcorr=False):
    # generate set of all pairs
    xcor_offset = 0 if auto_xcorr else 1
    event_inner = set(event1) & set(event2)
    event_outer1 = [evt for evt in event1 if evt not in event_inner]
    event_outer2 = [evt for evt in event2 if evt not in event_inner]
    event_inner = sorted(list(event_inner))
    pairs = {*()}
    if len(event_inner) > 0:
        pairs.update(
            set([(evt1, evt2) for i1, evt1 in enumerate(event_inner) for evt2 in event_inner[i1 + xcor_offset :]])
        )
        if len(event_outer1) > 0:
            pairs.update(set([(evt1, evt2) for evt1 in event_outer1 for evt2 in event_inner]))
        if len(event_outer2) > 0:
            pairs.update(set([(evt1, evt2) for evt1 in event_inner for evt2 in event_outer2]))
    if len(event_outer1) > 0 and len(event_outer2) > 0:
        pairs.update(set([(evt1, evt2) for evt1 in event_outer1 for evt2 in event_outer2]))
    return pairs


def read_pair_list(file_pair_list):
    # read pair ids from a text file
    pairs_df = pd.read_csv(file_pair_list, header=None, names=["event1", "event2"])
    pair_list = {(x["event1"], x["event2"]) for _, x in pairs_df.iterrows()}
    data_list1 = sorted(list(set(pairs_df["event1"].tolist())))
    data_list2 = sorted(list(set(pairs_df["event2"].tolist())))
    return pair_list, data_list1, data_list2


def generate_block_index(group1, group2, pair_list, min_sample_per_index=1):
    block_index = [(i, j) for i in range(len(group1)) for j in range(len(group2))]
    num_empty_index = []
    for i, j in block_index:
        num_samples = 0
        event1, event2 = group1[i], group2[j]
        for ii in range(len(event1)):
            for jj in range(len(event2)):
                if (event1[ii], event2[jj]) not in pair_list:
                    continue
                num_samples += 1
        if num_samples > min_sample_per_index:
            num_empty_index.append((i, j))
    return num_empty_index


def read_data(event, data_path, format="h5"):
    if format == "h5":
        data_list, info_list = read_das_eventphase_data_h5(
            data_path / f"{event}.h5", phase="P", event=True, dataset_keys=["shift_index"]
        )
    return {"data": data_list[0], "info": info_list[0]}


# helper reading functions
def read_das_eventphase_data_h5(fn, phase=None, event=False, dataset_keys=None, attrs_only=False):
    """
    read event phase data from hdf5 file
    Args:
        fn:  hdf5 filename
        phase: phase name list, e.g. ['P', 'S']
        dataset_keys: event phase data attributes, e.g. ['snr', 'traveltime', 'shift_index']
        event: if True, return event dict in info_list[0]
    Returns:
        data_list: list of event phase data
        info_list: list of event phase info
    """
    if isinstance(phase, str):
        phase = [phase]
    data_list = []
    info_list = []
    with h5py.File(fn, "r") as fid:
        g_phases = fid["data"]
        phase_avail = g_phases.keys()
        if phase is None:
            phase = list(phase_avail)
        for phase_name in phase:
            if not phase_name in g_phases.keys():
                raise (f"{fn} does not have phase: {phase_name}")
            g_phase = g_phases[phase_name]
            if attrs_only:
                data = []
            else:
                data = g_phase["data"][:]
            info = {}
            for key in g_phase["data"].attrs.keys():
                info[key] = g_phases[phase_name]["data"].attrs[key]
            if dataset_keys is not None:
                for key in dataset_keys:
                    if key in g_phase.keys():
                        info[key] = g_phase[key][:]
                        for kk in g_phase[key].attrs.keys():
                            info[kk] = g_phase[key].attrs[kk]
            data_list.append(data)
            info_list.append(info)
        if event:
            event_dict = dict((key, fid["data"].attrs[key]) for key in fid["data"].attrs.keys())
            info_list[0]["event"] = event_dict
    return data_list, info_list
