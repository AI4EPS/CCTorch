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
        pair_list,
        generate_pair,
        auto_xcor,
        block_num1,
        block_num2,
        data_path,
        shared_dict,
        device="cpu",
        transform=None,
        rank=0,
        world_size=1,
        **kwargs,
    ):
        super(CCIterableDataset).__init__()
        self.generate_pair = generate_pair
        self.auto_xcor = auto_xcor
        self.pairs, self.data_list1, self.data_list2 = self._read_pair(pair_list)
        self.group1 = [list(x) for x in np.array_split(self.data_list1, block_num1) if len(x) > 0]
        self.group2 = [list(x) for x in np.array_split(self.data_list2, block_num2) if len(x) > 0]
        self.block_num1 = len(self.group1)
        self.block_num2 = len(self.group2)
        # if len(self.group1) > len(self.group2):
        #     self.group1, self.group2 = self.group2, self.group1
        #     self.block_size1, self.block_size2 = self.block_size2, self.block_size1
        self.block_index = [(i, j) for i in range(len(self.group1)) for j in range(len(self.group2))][rank::world_size]
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
        # return iter(self.sample(self.block_index))
        return iter(self.sample(self.block_index[worker_id::num_workers]))

    def _read_pair(self, file_pair_list):
        # read pair ids from a text file
        df_pair = pd.read_csv(file_pair_list, header=None)
        ncol = len(df_pair.columns)
        if ncol == 1:
            # if the text file contains only one column, generate all possible pairs
            df_pair.rename(columns={0: "event1"}, inplace=True)
            data_list1 = sorted(list(set(df_pair["event1"].tolist())))
            data_list2 = data_list1
            pairs = self._generate_pair_set(data_list1, data_list2)
        elif ncol == 2:
            df_pair.rename(columns={0: "event1", 1: "event2"}, inplace=True)
            data_list1 = sorted(list(set(df_pair["event1"].tolist())))
            data_list2 = sorted(list(set(df_pair["event2"].tolist())))
            if self.generate_pair:
                # if the text file contains two columns and generate=True, generate all possible pairs
                pairs = self._generate_pair_set(data_list1, data_list2)
            else:
                # if the text file contains two columns and generate=False, only use pairs in the text file
                pairs = set((row["event1"], row["event2"]) for _, row in df_pair.iterrows())
        return pairs, data_list1, data_list2

    def _generate_pair_set(self, event1, event2):
        # generate set of all pairs
        if self.auto_xcor:
            xcor_offset = 0
        else:
            xcor_offset = 1
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

    def _read_das(self, event, local_dict):

        if event not in local_dict:
            data_list, info_list = read_das_eventphase_data_h5(
                self.data_path / f"{event}.h5", phase="P", event=True, dataset_keys=["shift_index"]
            )
            data = torch.tensor(data_list[0], device=self.device)
            info = info_list[0]
            if self.transform is not None:
                local_dict[event] = (self.transform(data), info)

        return local_dict[event]

    def sample(self, block_index):
        for i, j in block_index:
            local_dict = {}
            # if len(self.group1[i]) > len(self.group2[j]):
            #     event1, event2 = self.group2[j], self.group1[i]
            # else:
            #     event1, event2 = self.group1[i], self.group2[j]
            event1, event2 = self.group1[i], self.group2[j]
            # print(f"{len(event1) = }, {len(event2) = }")
            for ii in range(len(event1)):
                # begin = ii if i == j else 0
                for jj in range(len(event2)):
                    if (event1[ii], event2[jj]) not in self.pairs:
                        continue
                    # print(f"{i=}, {j=}, {ii=}, {jj=} {event1[ii]=}, {event2[jj]=}")
                    data_tuple1 = self._read_das(event1[ii], local_dict)
                    data_tuple2 = self._read_das(event2[jj], local_dict)
                    yield {
                        "event": event1[ii],
                        "data": data_tuple1[0],
                        "event_time": data_tuple1[1]["event"]["event_time"],
                        "shift_index": data_tuple1[1]["shift_index"],
                    }, {
                        "event": event2[jj],
                        "data": data_tuple2[0],
                        "event_time": data_tuple2[1]["event"]["event_time"],
                        "shift_index": data_tuple2[1]["shift_index"],
                    }

            del local_dict
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def __len__(self):
        # short_list = min(len(self.data_list1), len(self.data_list2))
        # return len(self.data_list1) * len(self.data_list2) - short_list * (short_list + 1) // 2
        return len(self.pairs)


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
