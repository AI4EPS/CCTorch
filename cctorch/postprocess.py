import numpy as np
import pandas as pd
import torch
import h5py
from datetime import datetime, timedelta

def write_xcor_data_to_h5(result, path_result, phase1='P', phase2='P'):
    """
    Write full xcor to hdf5 file. No reduce in time and channel axis
    """
    nbatch = result["xcor"].shape[0]
    dt = result["dt"]
    xcor = result["xcor"].cpu().numpy()
    channel_shift = int(result["channel_shift"])
    for ibatch in range(nbatch):
        id1 = int(result["id1"][ibatch].cpu().numpy())
        id2 = int(result["id2"][ibatch].cpu().numpy())
        fn = f"{path_result}/{id1}_{id2}.h5"
        xcor_dict = {'event_id1': id1,
                    'event_id2': id2,
                    'dt': dt,
                    'channel_shift': channel_shift,
                    'phase1': phase1,
                    'phase2': phase2}
        write_h5(fn, 'data', xcor[ibatch, :, :], xcor_dict)

def write_xcor_mccc_pick_to_csv(result, x, path_result, dt=0.01, channel_index=None):
    """
    """
    event_id1 = x[0]["event"].cpu().numpy()
    event_id2 = x[1]["event"].cpu().numpy()
    cc_main_lobe = result["cc_main"].cpu().numpy()
    cc_side_lobe = result["cc_side"].cpu().numpy()
    cc_dt = result["cc_dt"].cpu().numpy()
    fn_save = f"{path_result}/{event_id1}_{event_id2}.csv"
    phase_time1 = (datetime.fromisoformat(x[0]["event_time"]) + timedelta(seconds=dt) * x[0]["shift_index"].numpy())
    phase_time2 = (datetime.fromisoformat(x[1]["event_time"]) + timedelta(seconds=dt) * x[1]["shift_index"].numpy())
    if channel_index is None:
        channel_index = np.arange(len(cc_dt))
    pd.DataFrame({"channel_index": channel_index, 
                "event_id1": event_id1,
                "event_id2": event_id2,
                "phase_time1": [t.isoformat() for t in phase_time1],
                "phase_time2": [t.isoformat() for t in phase_time2],
                "cc_dt": cc_dt,
                "cc_main": cc_main_lobe,
                "cc_side": cc_side_lobe
                }).to_csv(fn_save, index=False)

def write_xcor_to_csv(result, path_result):
    """
    Write xcor to csv file. Reduce in time axis
    """
    nbatch = result["cc"].shape[0]
    cc = result["cc"].cpu().numpy()
    cc_dt = result["cc_dt"].cpu().numpy()
    for ibatch in range(nbatch):
        id1 = int(result["id1"][ibatch].cpu().numpy())
        id2 = int(result["id2"][ibatch].cpu().numpy())
        fn = f"{path_result}/{id1}_{id2}.csv"
        pd.DataFrame({"cc": cc[ibatch, :], "dt": cc_dt[ibatch, :]}).to_csv(fn, index=False)

def write_xcor_to_ccmat(result, ccmat, id_row, id_col):
    """
    Write single xcor value to a matrix. Reduce in both time and channel axis
    """
    nbatch = result["xcor"].shape[0]
    for ibatch in range(nbatch):
        """"""
        id1 = result["id1"][ibatch]
        id2 = result["id2"][ibatch]
        irow = torch.where(id_row == id1)
        icol = torch.where(id_col == id2)
        ccmat[irow, icol] = result["cc_mean"]

# helper functions
def write_h5(fn, dataset_name, data, attrs_dict):
    """
    write dataset to hdf5 file
    """
    with h5py.File(fn, "a") as fid:
        if dataset_name in fid.keys():
            del fid[dataset_name]
        fid.create_dataset(dataset_name, data=data)
        for key, val in attrs_dict.items():
            fid[dataset_name].attrs.modify(key, val)