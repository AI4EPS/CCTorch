import gcsfs
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
from obspy.signal.regression import linear_regression
from concurrent.futures import ProcessPoolExecutor, as_completed

from args import parse_args
import importlib

args = parse_args()

def scan_avai_pair_date_cloud(pair_dir):
    fs = gcsfs.GCSFileSystem()
    if pair_dir.startswith("gs://"):
        prefix = pair_dir
    else:
        prefix = f"gs://{pair_dir}"
    try:
        file_date_lst = fs.ls(prefix)
    except Exception as e:
        print(f"Failed to list {pair_dir}: {e}")
        return []
    date_lst = []
    for mwcs_file in file_date_lst:
        try:
            date_str = mwcs_file.split('/')[-1].split('.')[0]
            date_lst.append(date_str)
        except Exception as e:
            print(f"Failed to process {mwcs_file}: {e}")
            continue

    print(f"{prefix}: Found {len(date_lst)} dates")
    return sorted(list(set(date_lst)))

def wavg_wstd(data, errors):
    d = data
    errors[errors == 0] = 1e-6

    w = 1. / errors
    wavg = (d * w).sum() / w.sum()
    N = len(np.nonzero(w)[0])
    wstd = np.sqrt(np.sum(w * (d - wavg) ** 2) / ((N - 1) * np.sum(w) / N))
    return wavg, wstd

def process_pair(pair_dir: str, config, dtt_minlag_data, comp = 'ZZ', dtt_minlag_default_mode = False):
    file_dtt_out = config["file_dtt_out"]
    dtt_minlag_default = config["dtt_minlag_default"]
    mov_stacks = config["mov_stacks"]
    dtt_width = config["dtt_width"]
    dtt_sides = config["dtt_sides"]
    dtt_mincoh = config["dtt_mincoh"]
    dtt_maxerr = config["dtt_maxerr"]
    dtt_maxdt = config["dtt_maxdt"]
    filterid = config["filterid"]
    time_type = config["time_type"]

    pair_name = pair_dir.split('/')[-1]
    # print(f"=== Start pair_dir: {pair_dir} ===")

    date_available = scan_avai_pair_date_cloud(pair_dir)
    if not date_available:
        print(f"No available dates for pair_dir: {pair_dir}, skipping.")
        return
    
    fs_local = gcsfs.GCSFileSystem()
    if not fs_local.exists(file_dtt_out):
            fs_local.makedirs(file_dtt_out, exist_ok=True)

    dtt_minlag = dtt_minlag_data.get(pair_name, dtt_minlag_default)
    if dtt_minlag_default_mode:
        dtt_minlag = dtt_minlag_default
    if pair_name not in dtt_minlag_data:
        print(f"Using default dtt_minlag {dtt_minlag_default} for pair {pair_name}")
    
    comp_lst_dtt = [comp]
    for components in comp_lst_dtt:
        for current in date_available:
            for mov_stack in mov_stacks:
            
                first = True
                if pair_dir.startswith("gs://"):
                    day = os.path.join(pair_dir, f"{current}.txt")
                else:
                    day = os.path.join("gs://", pair_dir, f"{current}.txt")

                df = pd.read_csv(
                    day, delimiter=' ', header=None, index_col=0,
                    names=['t', 'dt', 'err', 'coh'])
                
                tArray = df.index.values
                lmlag = -dtt_minlag
                rmlag = dtt_minlag
                lMlag = lmlag - dtt_width
                rMlag = rmlag + dtt_width

                if dtt_sides == "both":
                    tindex = np.where(((tArray >= lMlag) & (tArray <= lmlag)) | ((tArray >= rmlag) & (tArray <= rMlag)))[0]
                elif dtt_sides == "left":
                    tindex = np.where((tArray >= lMlag) & (tArray <= lmlag))[0]
                else:
                    tindex = np.where((tArray >= rmlag) & (tArray <= rMlag))[0]

                tmp = np.setdiff1d(np.arange(len(tArray)),tindex)
                df.loc[df.index[tmp], 'err'] = 1.0
                df.loc[df.index[tmp], 'coh'] = 0.0

                if first:
                    tArray = df.index.values
                    dtArray = df['dt']
                    errArray = df['err']
                    cohArray = df['coh']
                    pairArray = [pair_name, ]
                    first = False
                else:
                    dtArray = np.vstack((dtArray, df['dt']))
                    errArray = np.vstack((errArray, df['err']))
                    cohArray = np.vstack((cohArray, df['coh']))
                    pairArray.append(pair_name)
                del df
                del day
                if not first:
                    #~ tindex = np.tindwhere(((tArray >= lMlag) & (tArray <= lmlag)) | (
                        #~ (tArray >= rmlag) & (tArray <= rMlag)))[0]

                    Dates = []
                    Pairs = []
                    M = []
                    EM = []
                    A = []
                    EA = []
                    M0 = []
                    EM0 = []
                    if len(pairArray) != 1:
                        # first stack all pairs to a ALL mean pair, using
                        # indexes of selected values:
                        new_dtArray = np.zeros(len(tArray))
                        new_errArray = np.zeros(len(tArray)) + 9999
                        new_cohArray = np.zeros(len(tArray))
                        for i in range(len(tArray)):
                            #~ if i in tindex:
                            if 1:
                                cohindex = np.where(
                                    cohArray[:, i] >= dtt_mincoh)[0]
                                errindex = np.where(
                                    errArray[:, i] <= dtt_maxerr)[0]
                                dtindex = np.where(
                                    np.abs(dtArray[:, i]) <= dtt_maxdt)[0]
                                index = np.intersect1d(cohindex, errindex)
                                index = np.intersect1d(index, dtindex)

                                wavg, wstd = wavg_wstd(
                                    dtArray[:, i][index],
                                    errArray[:, i][index])
                                new_dtArray[i] = wavg
                                new_errArray[i] = wstd
                                new_cohArray[i] = 1.0

                        dtArray = np.vstack((dtArray, new_dtArray))
                        errArray = np.vstack((errArray, new_errArray))
                        cohArray = np.vstack((cohArray, new_cohArray))
                        pairArray.append("ALL")
                        del new_cohArray, new_dtArray, new_errArray,\
                            cohindex, errindex, dtindex, wavg, wstd
                        
                        # then stack selected pais to GROUPS:
                        groups = {}
                        npairs = len(pairArray)-1
                        for group in groups.keys():
                            new_dtArray = np.zeros(len(tArray))
                            new_errArray = np.zeros(len(tArray)) + 9999
                            new_cohArray = np.zeros(len(tArray))
                            pairindex = []
                            for j, pair in enumerate(pairArray[:npairs]):
                                net1, sta1, net2, sta2 = pair.split('_')
                                if sta1 in groups[group] and \
                                                sta2 in groups[group]:
                                    pairindex.append(j)
                            pairindex = np.array(pairindex)

                            for i in range(len(tArray)):
                                #~ if i in tindex:
                                if 1:
                                    cohindex = np.where(
                                        cohArray[:, i] >= dtt_mincoh)[0]
                                    errindex = np.where(
                                        errArray[:, i] <= dtt_maxerr)[0]
                                    dtindex = np.where(
                                        np.abs(dtArray[:, i]) <= dtt_maxdt)[0]
                                    

                                    index = np.intersect1d(cohindex,
                                                            errindex)
                                    index = np.intersect1d(index, dtindex)
                                    index = np.intersect1d(index, pairindex)
                                    

                                    wavg, wstd = wavg_wstd(
                                        dtArray[:, i][index],
                                        errArray[:, i][index])
                                    new_dtArray[i] = wavg
                                    new_errArray[i] = wstd
                                    new_cohArray[i] = 1.0

                            dtArray = np.vstack((dtArray, new_dtArray))
                            errArray = np.vstack((errArray, new_errArray))
                            cohArray = np.vstack((cohArray, new_cohArray))
                            pairArray.append(group)
                            del new_cohArray, new_dtArray, new_errArray,\
                                cohindex, errindex, dtindex, wavg, wstd
                            # END OF GROUP HANDLING

                    # then process all pairs + the ALL
                    if len(dtArray.shape) == 1:  # if there is only one pair:
                        dtArray = dtArray.values.reshape((1, dtArray.shape[0]))
                        cohArray = cohArray.values.reshape((1, cohArray.shape[0]))
                        errArray = errArray.values.reshape((1, errArray.shape[0]))

                    used = np.zeros(dtArray.shape)

                    for i, pair in enumerate(pairArray):
                        cohindex = np.where(cohArray[i] >= dtt_mincoh)[0]
                        errindex = np.where(errArray[i] <= dtt_maxerr)[0]
                        dtindex = np.where(np.abs(dtArray[i]) <= dtt_maxdt)[0]

                        #~ index = np.intersect1d(tindex, cohindex)
                        index = np.intersect1d(cohindex, errindex)
                        index = np.intersect1d(index, dtindex)

                        used[i][index] = 1.0

                        w = 1.0 / errArray[i][index]
                        w[~np.isfinite(w)] = 1.0
                        VecXfilt = tArray[index]
                        VecYfilt = dtArray[i][index]
                        if len(VecYfilt) >= 2:
                            m, a, em, ea = linear_regression(
                                VecXfilt, VecYfilt, w,
                                intercept_origin=False)
                            m0, em0 = linear_regression(
                                VecXfilt, VecYfilt, w,
                                intercept_origin=True)
                            M.append(m)
                            EM.append(em)
                            A.append(a)
                            EA.append(ea)

                            M0.append(m0)
                            EM0.append(em0)

                            Dates.append(current)
                            Pairs.append(pair)

                            del m, a, em, ea, m0, em0

                        del VecXfilt, VecYfilt, w
                        del index, cohindex, errindex, dtindex

                    # logger.debug(
                    #     "%s: exporting: %i pairs" % (current,
                    #                                     len(pairArray)))
                    df_out = pd.DataFrame(
                        {'Pairs': Pairs, 'M': M, 'EM': EM, 'A': A, 'EA': EA,
                            'M0': M0, 'EM0': EM0},
                        index=Dates)
                    # Needs to be changed !
                    output = os.path.join(file_dtt_out,
                        'DTT', "%02i" % filterid, f"%03i_{time_type}" % mov_stack,
                        components, pair_name)
                    # print(f"Saving DTT results to {output}")
                    if not fs_local.exists(output):
                        fs_local.makedirs(output, exist_ok=True)
                    fn = os.path.join(output, '%s.txt' % current)
                    if fs_local.exists(fn):
                        with fs_local.open(fn, 'rt') as f_in:
                            existing = pd.read_csv(f_in, index_col="Pairs", parse_dates=True)
                        
                        for _, row in df_out.iterrows():
                            if row.Pairs in existing.index.values:
                                existing.drop(row.Pairs, inplace=True)
                                # logger.debug("Pair: %s is already in the output file, overwriting" % row.Pairs)
                        existing["Pairs"] = existing.index.values
                        existing.set_index("Date", inplace=True)
                        df_out = pd.concat([df_out, existing])

                    with fs_local.open(fn, 'wt') as gcs_file:
                        df_out.to_csv(gcs_file, index_label='Date')
                    # print(f"{pair_dir}: DTT results saved to {fn}")

                    del df_out, M, EM, A, EA, M0, EM0, Pairs, Dates, used
                    del tArray, dtArray, errArray, cohArray, pairArray
                    del output


if __name__ == "__main__":
    node_rank = args.node_rank
    num_nodes = args.num_nodes

    cfg = importlib.import_module(f"configs.{args.project}")
    component_mwcs = cfg.component_mwcs
    folder_mwcs = cfg.folder_mwcs
    # file_list_path = cfg.mwcs_paths
    file_lst_dtt = cfg.file_dtt_minlag

    config = {"file_dtt_out": cfg.folder_dtt,
              "dtt_minlag_default": cfg.dtt_minlag_default,
              "mov_stacks": cfg.mov_stacks,
              "dtt_width": cfg.dtt_width,
              "dtt_sides": cfg.dtt_sides,
              "dtt_mincoh": cfg.dtt_mincoh,
              "dtt_maxerr": cfg.dtt_maxerr,
              "dtt_maxdt": cfg.dtt_maxdt,
              "filterid": cfg.filterid_dtt,
              "time_type": cfg.time_type}
    
    fs = gcsfs.GCSFileSystem()
    fs.invalidate_cache("gs://")

    path_mwcs = f'{folder_mwcs}MWCS/01/001_{cfg.time_type}/{component_mwcs}'
    pair_dirs = fs.ls(path_mwcs)

    # with fs.open(file_list_path, "r") as f:
    #     pair_dirs = [line.strip() for line in f.readlines()]
    print(f"Total pair_dirs found: {len(pair_dirs)} from {path_mwcs}")

    with fs.open(file_lst_dtt, "r") as f:
        dtt_minlag_data = {line.strip().split()[0]: float(line.strip().split()[1]) for line in f.readlines()}


    pair_dirs = pair_dirs[node_rank::num_nodes]

    max_workers = min(len(pair_dirs), os.cpu_count() or 4)
    print(f"Using {max_workers} workers for {len(pair_dirs)} pair_dirs with component {component_mwcs} on node_rank {node_rank}/{num_nodes}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pair, p, config, dtt_minlag_data, component_mwcs, dtt_minlag_default_mode=cfg.dynamic_dtt_minlag): p for p in pair_dirs}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"Error in pair_dir {p}: {e}")

    print(f"Node rank {node_rank} completed processing {len(pair_dirs)} pair_dirs.")