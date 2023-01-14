from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.signal
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, IterableDataset
import itertools


class CCDataset(Dataset):
    def __init__(
        self,
        config=None,
        pair_list=None,
        data_list1=None,
        data_list2=None,
        data_path="./",
        data_format="h5",
        block_num1=1,
        block_num2=1,
        device="cpu",
        transforms=None,
        rank=0,
        world_size=1,
        **kwargs,
    ):
        super(CCDataset).__init__()
        ## TODO: extract this part into a function; keep this temporary until TM and AM are implemented
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
                self.pair_list = generate_pairs(self.data_list1, self.data_list2, config.auto_xcorr)

        self.mode = config.mode
        self.config = config
        self.block_num1 = block_num1
        self.block_num2 = block_num2
        self.group1 = [list(x) for x in np.array_split(self.data_list1, block_num1) if len(x) > 0]
        self.group2 = [list(x) for x in np.array_split(self.data_list2, block_num2) if len(x) > 0]
        self.block_index = generate_block_index(self.group1, self.group2, self.pair_list)
        self.data_path = Path(data_path)
        self.data_format = data_format
        self.transforms = transforms
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

            if self.transforms is not None:
                data = self.transforms(data)
        else:
            data = torch.empty((1, 1, 1), dtype=torch.float32).to(self.device)
            pair_index = torch.tensor([[0, 0]], dtype=torch.int64).to(self.device)

        return {"data": data, "info": info, "pair_index": pair_index}

    def __len__(self):
        return len(self.block_index)


class CCIterableDataset(IterableDataset):
    def __init__(
        self,
        config=None,
        pair_list=None,
        data_list1=None,
        data_list2=None,
        data_path="./",
        data_format="h5",
        block_num1=1,
        block_num2=1,
        device="cpu",
        transforms=None,
        batch_size=32,
        rank=0,
        world_size=1,
        **kwargs,
    ):
        super(CCIterableDataset).__init__()
        ## TODO: extract this part into a function; keep this temporary until TM and AM are implemented
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
            if (pair_list is None) and (config.mode != "AM"):
                self.pair_list = generate_pairs(self.data_list1, self.data_list2, config.auto_xcorr)

        self.mode = config.mode
        self.config = config
        self.block_num1 = block_num1
        self.block_num2 = block_num2
        self.data_path = Path(data_path)
        self.data_format = data_format
        self.transforms = transforms
        self.batch_size = batch_size
        self.device = device

        if self.mode == "AM":
             ## For ambient noise, we split chunks in the sampling function
            self.data_list1 = self.data_list1[rank::world_size]
        else:
            self.group1 = [list(x) for x in np.array_split(self.data_list1, block_num1) if len(x) > 0]
            self.group2 = [list(x) for x in np.array_split(self.data_list2, block_num2) if len(x) > 0]
            self.block_index = generate_block_index(self.group1, self.group2, self.pair_list)[rank::world_size]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        if self.mode == "AM":
            return iter(self.sample_ambient_noise(self.data_list1[worker_id::num_workers]))
        else:
            return iter(self.sample(self.block_index[worker_id::num_workers]))

    def sample(self, block_index):

        for i, j in block_index:
            local_dict = {}
            event1, event2 = self.group1[i], self.group2[j]
            data1, info1, data2, info2 = [], [], [], []
            num = 0
            for (ii, jj) in itertools.product(range(len(event1)), range(len(event2))):
                if (event1[ii], event2[jj]) not in self.pair_list:
                    continue

                if event1[ii] not in local_dict:
                    meta1 = read_data(event1[ii], self.data_path, self.data_format)
                    data = torch.tensor(meta1["data"]).to(self.device).unsqueeze(0)  ## add batch dimension
                    if self.transforms is not None:
                        data = self.transforms(data)
                    meta1["data"] = data
                    local_dict[event1[ii]] = meta1
                else:
                    meta1 = local_dict[event1[ii]]

                if event2[jj] not in local_dict:
                    meta2 = read_data(event2[jj], self.data_path, self.data_format)
                    data = torch.tensor(meta2["data"]).to(self.device).unsqueeze(0)  ## add batch dimension
                    if self.transforms is not None:
                        data = self.transforms(data)
                    meta1["data"] = data
                    local_dict[event2[jj]] = meta2
                else:
                    meta2 = local_dict[event2[jj]]

                data1.append(meta1["data"])
                info1.append(meta1["info"])
                data2.append(meta2["data"])
                info2.append(meta2["info"])

                num += 1
                if num == self.batch_size:
                    num = 0
                    data_batch1 = torch.cat(data1, dim=0)
                    data_batch2 = torch.cat(data2, dim=0)
                    ## TODO: fix yield for batch size
                    info_batch1 = None
                    info_batch2 = None
                    data1, info1, data2, info2 = [], [], [], []
                    yield {"data": data_batch1, "info": info_batch1}, {"data": data_batch2, "info": info_batch2}

            ## yield the last batch
            if num > 0:
                data_batch1 = torch.cat(data1, dim=0)
                data_batch2 = torch.cat(data2, dim=0)
                ## TODO: fix yield for batch size
                info_batch1 = None
                info_batch2 = None
                data1, info1, data2, info2 = [], [], [], []
                yield {"data": data_batch1, "info": info_batch1}, {"data": data_batch2, "info": info_batch2}

            # del local_dict
            # if self.device == "cuda":
            #     torch.cuda.empty_cache()

    def sample_ambient_noise(self, data_list):

        for fd in data_list:

            meta = read_data(fd, self.data_path, self.data_format, mode=self.mode)  # (nch, nt)
            data = meta["data"].unsqueeze(0)  # (1, nch, nt)
            
            if (self.config.transforms_on_file) and (self.transforms is not None):
                data = self.transforms(data)

            nbatch, nch, nt = data.shape

            ## cut blocks
            min_channel = self.config.min_channel if self.config.min_channel is not None else 0
            max_channel = self.config.max_channel if self.config.max_channel is not None else nch
            left_end_channel = self.config.left_end_channel if self.config.left_end_channel is not None else -nch
            right_end_channel = self.config.right_end_channel if self.config.right_end_channel is not None else nch

            if self.config.fixed_channels is not None:
                ## only process channels passed by "--fixed-channels" as source
                lists_1 = (
                    self.config.fixed_channels
                    if isinstance(self.config.fixed_channels, list)
                    else [self.fixed_channels]
                )
            else:
                ## using delta_channel to down-sample channels needed for ambient noise
                ## using min_channel and max_channel to selected channels that are within a range 
                lists_1 = range(min_channel, max_channel, self.config.delta_channel)
            lists_2 = range(min_channel, max_channel, self.config.delta_channel)
            group_1 = [list(x) for x in np.array_split(lists_1, self.block_num1) if len(x) > 0]
            group_2 = [list(x) for x in np.array_split(lists_2, self.block_num2) if len(x) > 0]
            block_index = list(itertools.product(range(len(group_1)), range(len(group_2))))
            
            ## loop each block
            for i, j in block_index:
                block1 = group_1[i]
                block2 = group_2[j]
                index_i = []
                index_j = []
                for ii, jj in itertools.product(block1, block2):
                    if (jj < (ii + left_end_channel)) or (jj > (ii + right_end_channel)):
                        continue
                    index_i.append(ii)
                    index_j.append(jj)

                data_i = data[:, index_i, :].to(self.device)
                data_j = data[:, index_j, :].to(self.device)
                
                if (self.config.transforms_on_batch) and (self.transforms is not None):
                    data_i = self.transforms(data_i)
                    data_j = self.transforms(data_j)

                yield {"data": data_i, "info": {"index": index_i},},  {"data": data_j, "info": {"index": index_j}}


    def __len__(self):

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        if self.mode == "AM":
            num_samples = self.count_sample_ambient_noise(num_workers, worker_id)
        else:
            num_samples = self.count_sample(num_workers, worker_id)

        return num_samples

    def count_sample(self, num_workers, worker_id):
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

    def count_sample_ambient_noise(self, num_workers, worker_id):
        num_samples = 0
        for fd in self.data_list1:

            nch, nt = get_shape_das_continous_data_h5(self.data_path / fd)  # (nch, nt)

            ## cut blocks
            min_channel = self.config.min_channel if self.config.min_channel is not None else 0
            max_channel = self.config.max_channel if self.config.max_channel is not None else nch
            left_end_channel = self.config.left_end_channel if self.config.left_end_channel is not None else -nch
            right_end_channel = self.config.right_end_channel if self.config.right_end_channel is not None else nch

            if self.config.fixed_channels is not None:
                ## only process channels passed by "--fixed-channels" as source
                lists_1 = (
                    self.config.fixed_channels
                    if isinstance(self.config.fixed_channels, list)
                    else [self.fixed_channels]
                )
            else:
                ## using delta_channel to down-sample channels needed for ambient noise
                ## using min_channel and max_channel to selected channels that are within a range 
                lists_1 = range(min_channel, max_channel, self.config.delta_channel)
            lists_2 = range(min_channel, max_channel, self.config.delta_channel)
            group_1 = [list(x) for x in np.array_split(lists_1, self.block_num1) if len(x) > 0]
            group_2 = [list(x) for x in np.array_split(lists_2, self.block_num2) if len(x) > 0]
            block_index = list(itertools.product(range(len(group_1)), range(len(group_2))))
            
            ## loop each block
            for i, j in block_index:
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


def generate_block_index(group1, group2, pair_list, min_sample_per_block=1):
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
        if num_samples > min_sample_per_block:
            num_empty_index.append((i, j))
    return num_empty_index


def read_data(file_name, data_path, format="h5", mode="CC"):

    if (format == "h5") and (mode == "CC"):
        data_list, info_list = read_das_eventphase_data_h5(
            data_path / file_name, phase="P", event=True, dataset_keys=["shift_index"]
        )
        ## TODO: check with Jiaxuan; why do we need to read a list but return the first one
        data = data_list[0]
        info = info_list[0]

    if (format == "h5") and (mode == "AM"):
        data, info = read_das_continuous_data_h5(data_path / file_name, dataset_keys=[])

    return {"data": torch.tensor(data), "info": info}


def read_das_continuous_data_h5(fn, dataset_keys=[]):
    with h5py.File(fn, "r") as f:
        data = f["Data"][:]
        info = {}
        for key in dataset_keys:
            info[key] = f[key][:]
    return data, info

def get_shape_das_continous_data_h5(file):
    with h5py.File(file, "r") as f:
        data_shape = f["Data"].shape
    return data_shape


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
