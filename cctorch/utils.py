# %%
import math
import multiprocessing as mp
import os
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from multiprocessing import shared_memory
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

import gamma
import h5py
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import scipy
import scipy.signal
import torch
from tqdm.auto import tqdm
import json

# %%
def write_results(results, result_path, ccconfig, rank=0, world_size=1):
    if ccconfig.mode == "CC":
        ## TODO: add writting for CC
        write_cc_pairs(results, result_path, ccconfig, rank=rank, world_size=world_size)
    elif ccconfig.mode == "TM":
        ## TODO: add writting for CC
        pass
    elif ccconfig.mode == "AN":
        write_ambient_noise(results, result_path, ccconfig, rank=rank, world_size=world_size)
    else:
        raise ValueError(f"{ccconfig.mode} not supported")


def write_cc_pairs(results, result_path, ccconfig, rank=0, world_size=1, plot_figure=False):
    """
    Write cross-correlation results to disk.
    Parameters
    ----------
    results : list of dict
        List of results from cross-correlation. 
        e.g. [{
            "topk_index": topk_index, 
            "topk_score": topk_score, 
            "neighbor_score": neighbor_score, 
            "pair_index": pair_index}]
    """
    
    if not isinstance(result_path, Path):
        result_path = Path(result_path)

    min_cc_score = ccconfig.min_cc_score
    min_cc_ratio = ccconfig.min_cc_ratio  
    min_cc_weight = ccconfig.min_cc_weight  
    
    if "cc_mean" in results[0]:
        with open(result_path / f"{ccconfig.mode}_{rank:03d}_{world_size:03d}.txt", "a") as fp:
            for meta in results:
                cc_mean = meta["cc_mean"]
                pair_index = meta["pair_index"]
                nb, nch = cc_mean.shape
                for i in range(nb):
                    pair_id = pair_index[i]
                    id1, id2 = pair_id
                    score = ','.join([f"{x.item():.3f}" for x in cc_mean[i]])
                    fp.write(f"{id1},{id2},{score}\n")

    with h5py.File(result_path / f"{ccconfig.mode}_{rank:03d}_{world_size:03d}.h5", "a") as fp:
        for meta in results:
            topk_index = meta["topk_index"]
            topk_score = meta["topk_score"]
            neighbor_score = meta["neighbor_score"]
            pair_index = meta["pair_index"]

            nb, nch, nx, nk = topk_index.shape

            for i in range(nb):

                cc_score = topk_score[i, :, :, 0]
                cc_weight = topk_score[i, :, :, 0] - topk_score[i, :, :, 1]

                if ((cc_score.max() >= min_cc_score) and (cc_weight.max() >= min_cc_weight) and
                    (torch.sum((cc_score > min_cc_score) & (cc_weight > min_cc_weight)) >= nch * nx * min_cc_ratio)):

                    pair_id = pair_index[i]
                    id1, id2 = pair_id
                    if int(id1) > int(id2):
                        id1, id2 = id2, id1
                        topk_index = - topk_index
                
                    if f"{id1}/{id2}" not in fp:
                        gp = fp.create_group(f"{id1}/{id2}")
                    else:
                        gp = fp[f"{id1}/{id2}"]
                    
                    if f"cc_index" in gp:
                        del gp["cc_index"]
                    gp.create_dataset(f"cc_index", data=topk_index[i].cpu())
                    if f"cc_score" in gp:
                        del gp["cc_score"]
                    gp.create_dataset(f"cc_score", data=topk_score[i].cpu())
                    if f"cc_weight" in gp:
                        del gp["cc_weight"]
                    gp.create_dataset(f"cc_weight", data=cc_weight.cpu())
                    if f"neighbor_score" in gp:
                        del gp["neighbor_score"]
                    gp.create_dataset(f"neighbor_score", data=neighbor_score[i].cpu())
                    
                    if id2 != id1:
                        if f"{id2}/{id1}" not in fp:
                            # fp[f"{id2}/{id1}"] = h5py.SoftLink(f"/{id1}/{id2}")
                            gp = fp.create_group(f"{id2}/{id1}")
                        else:
                            gp = fp[f"{id2}/{id1}"]
                        
                        if f"cc_index" in gp:
                            del gp["cc_index"]
                        gp.create_dataset(f"cc_index", data= - topk_index[i].cpu())
                        if f"neighbor_score" in gp:
                            del gp["neighbor_score"]
                        gp.create_dataset(f"neighbor_score", data=neighbor_score[i].cpu().flip(-1))
                        if f"cc_score" in gp:
                            del gp["cc_score"]
                        gp["cc_score"] = fp[f"{id1}/{id2}/cc_score"]
                        if f"cc_weight" in gp:
                            del gp["cc_weight"]
                        gp["cc_weight"] = fp[f"{id1}/{id2}/cc_weight"]
                    
                    if plot_figure:
                        for j in range(nch):
                            fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(10, 20), sharey=True)
                            ax[0, 0].imshow(meta["xcorr"][i, j, :, :].cpu().numpy(), cmap="seismic", vmax=1, vmin=-1, aspect="auto")
                            for k in range(nx):
                                ax[0, 1].plot(meta["data1"][i, j, k, :].cpu().numpy()/np.max(np.abs(meta["data1"][i, j, k, :].cpu().numpy()))+k, linewidth=1, color="k")
                                ax[0, 2].plot(meta["data2"][i, j, k, :].cpu().numpy()/np.max(np.abs(meta["data2"][i, j, k, :].cpu().numpy()))+k, linewidth=1, color="k")
                            
                            try:
                                fig.savefig(f"debug/test_{pair_id[0]}_{pair_id[1]}_{j}.png", dpi=300)
                            except:
                                os.mkdir("debug")
                                fig.savefig(f"debug/test_{pair_id[0]}_{pair_id[1]}_{j}.png", dpi=300)
                            print(f"debug/test_{pair_id[0]}_{pair_id[1]}_{j}.png")
                            plt.close(fig)
                            

def write_ambient_noise(results, result_path, ccconfig, rank=0, world_size=1):

    if not isinstance(result_path, Path):
        result_path = Path(result_path)

    with h5py.File(result_path / f"{ccconfig.mode}_{rank:03d}_{world_size:03d}.h5", "a") as fp:
        for meta in results:
            xcorr = meta["xcorr"].cpu().numpy()
            nb, nch, nx, nt = xcorr.shape
            for i in range(nb):
                data = np.squeeze(np.nan_to_num(xcorr[i, :, :, :]))

                # for j, pair_id in enumerate(meta["pair_index"]):
                for pair_id in meta["pair_index"]:
                    list1, list2 = pair_id
                    
                    for j, (id1, id2) in enumerate(zip(list1, list2)):
                        if f"{id1}/{id2}" not in fp:
                            gp = fp.create_group(f"{id1}/{id2}")
                            ds = gp.create_dataset("xcorr", data=data[..., j, :])
                            ds.attrs["count"] = 1
                        else:
                            gp = fp[f"{id1}/{id2}"]
                            ds = gp["xcorr"]
                            count = ds.attrs["count"]
                            ds[:] = count / (count + 1) * ds[:] + data[..., j, :] / (count + 1)
                            ds.attrs["count"] = count + 1
                        
                        if f"{id2}/{id1}" not in fp:
                            gp = fp.create_group(f"{id2}/{id1}")
                            ds = gp.create_dataset("xcorr", data=np.flip(data[..., j, :], axis=-1))
                            ds.attrs["count"] = 1
                        else:
                            gp = fp[f"{id2}/{id1}"]
                            ds = gp["xcorr"]
                            count = ds.attrs["count"]
                            ds[:] = count / (count + 1) * ds[:] + np.flip(data[..., j, :], axis=-1) / (count + 1)
                            ds.attrs["count"] = count + 1


def write_xcor_data_to_h5(result, path_result, phase1="P", phase2="P"):
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
        xcor_dict = {
            "event_id1": id1,
            "event_id2": id2,
            "dt": dt,
            "channel_shift": channel_shift,
            "phase1": phase1,
            "phase2": phase2,
        }
        write_h5(fn, "data", xcor[ibatch, :, :], xcor_dict)


def write_xcor_mccc_pick_to_csv(result, x, path_result, dt=0.01, channel_index=None):
    """ """
    event_id1 = x[0]["event"].cpu().numpy()
    event_id2 = x[1]["event"].cpu().numpy()
    cc_main_lobe = result["cc_main"].cpu().numpy()
    cc_side_lobe = result["cc_side"].cpu().numpy()
    cc_dt = result["cc_dt"].cpu().numpy()
    fn_save = f"{path_result}/{event_id1}_{event_id2}.csv"
    phase_time1 = datetime.fromisoformat(x[0]["event_time"]) + timedelta(seconds=dt) * x[0]["shift_index"].numpy()
    phase_time2 = datetime.fromisoformat(x[1]["event_time"]) + timedelta(seconds=dt) * x[1]["shift_index"].numpy()
    if channel_index is None:
        channel_index = np.arange(len(cc_dt))
    pd.DataFrame(
        {
            "channel_index": channel_index,
            "event_id1": event_id1,
            "event_id2": event_id2,
            "phase_time1": [t.isoformat() for t in phase_time1],
            "phase_time2": [t.isoformat() for t in phase_time2],
            "cc_dt": cc_dt,
            "cc_main": cc_main_lobe,
            "cc_side": cc_side_lobe,
        }
    ).to_csv(fn_save, index=False)


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


def reduce_ccmat(file_cc_matrix, channel_shift, nrank, clean=True):
    """
    reduce the cc matrix calculated from different cores
    """
    for rank in range(nrank):
        data = np.load(f"{file_cc_matrix}_{channel_shift}_{rank}.npz")
        if rank == 0:
            cc = data["cc"]
            id_row = data["id_row"]
            id_col = data["id_col"]
            id_pair = data["id_pair"]
        else:
            cc += data["cc"]
    np.savez(f"{file_cc_matrix}_{channel_shift}.npz", cc=cc, id_row=id_row, id_col=id_col, id_pair=id_pair)
    if clean:
        for rank in range(nrank):
            os.remove(f"{file_cc_matrix}_{channel_shift}_{rank}.npz")


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
