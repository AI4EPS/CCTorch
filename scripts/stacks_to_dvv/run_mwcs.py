import os
from datetime import datetime, timedelta
import numpy as np
import zarr
from obspy import read
import gcsfs
from matplotlib import pyplot as plt

from obspy.signal.invsim import cosine_taper
from obspy.signal.regression import linear_regression
import scipy
import scipy.fft as sf

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging

from args import parse_args
import importlib

args = parse_args()

def nextpow2(x):
    return np.ceil(np.log2(np.abs(x)))

def smooth(x, window='boxcar', half_win=3):
    """ some window smoothing """
    # TODO: docsting
    window_len = 2 * half_win + 1
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    if window == "boxcar":
        w = scipy.signal.windows.boxcar(window_len).astype('complex')
    else:
        w = scipy.signal.windows.hann(window_len).astype('complex')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[half_win:len(y) - half_win]

def getCoherence(dcs, ds1, ds2):
    # TODO: docsting
    n = len(dcs)
    coh = np.zeros(n).astype('complex')
    valids = np.argwhere(np.logical_and(np.abs(ds1) > 0, np.abs(ds2 > 0)))
    coh[valids] = dcs[valids] / (ds1[valids] * ds2[valids])
    coh[coh > (1.0 + 0j)] = 1.0 + 0j
    return coh

def mwcs(current, reference, freqmin, freqmax, df, tmin, window_length, step,
         smoothing_half_win=5):
    
    delta_t = []
    delta_err = []
    delta_mcoh = []
    time_axis = []

    window_length_samples = int(window_length * df)
    padd = int(2 ** (nextpow2(window_length_samples) + 2))
    count = 0
    tp = cosine_taper(window_length_samples, 0.85)
    minind = 0
    maxind = window_length_samples
    while maxind <= len(current):
        cci = current[minind:(minind + window_length_samples)]
        cci = scipy.signal.detrend(cci, type='linear')
        cci *= tp

        cri = reference[minind:(minind + window_length_samples)]
        cri = scipy.signal.detrend(cri, type='linear')
        cri *= tp

        minind += int(step*df)
        maxind += int(step*df)

        fcur = sf.fft(cci, n=padd)[:padd // 2]
        fref = sf.fft(cri, n=padd)[:padd // 2]

        fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
        fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2

        # Calculate the cross-spectrum
        X = fref * (fcur.conj())
        if smoothing_half_win != 0:
            dcur = np.sqrt(smooth(fcur2, window='hanning',
                                  half_win=smoothing_half_win))
            dref = np.sqrt(smooth(fref2, window='hanning',
                                  half_win=smoothing_half_win))
            X = smooth(X, window='hanning',
                       half_win=smoothing_half_win)
        else:
            dcur = np.sqrt(fcur2)
            dref = np.sqrt(fref2)

        dcs = np.abs(X)

        # Find the values the frequency range of interest
        freq_vec = sf.fftfreq(len(X) * 2, 1. / df)[:padd // 2]
        index_range = np.argwhere(np.logical_and(freq_vec >= freqmin,
                                                 freq_vec <= freqmax))

        # Get Coherence and its mean value
        coh = getCoherence(dcs, dref, dcur)
        mcoh = np.mean(coh[index_range])

        # Get Weights
        w = 1.0 / (1.0 / (coh[index_range] ** 2) - 1.0)
        w[coh[index_range] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
        w = np.sqrt(w * np.sqrt(dcs[index_range]))
        w = np.real(w)

        # Frequency array:
        v = np.real(freq_vec[index_range]) * 2 * np.pi

        # Phase:
        phi = np.angle(X)
        phi[0] = 0.
        phi = np.unwrap(phi)
        phi = phi[index_range]

        # Calculate the slope with a weighted least square linear regression
        # forced through the origin
        # weights for the WLS must be the variance !
        m, em = linear_regression(v.flatten(), phi.flatten(), w.flatten())

        delta_t.append(m)

        e = np.sum((phi - m * v) ** 2) / (np.size(v) - 1)
        s2x2 = np.sum(v ** 2 * w ** 2)
        sx2 = np.sum(w * v ** 2)
        e = np.sqrt(e * s2x2 / sx2 ** 2)

        delta_err.append(e)
        delta_mcoh.append(np.real(mcoh))
        time_axis.append(tmin+window_length/2.+count*step)
        count += 1

        del fcur, fref
        del X
        del freq_vec
        del index_range
        del w, v, e, s2x2, sx2, m, em

    if maxind > len(current) + step*df:
        logging.warning("The last window was too small, but was computed")

    return np.array([time_axis, delta_t, delta_err, delta_mcoh]).T


def load_stacked_ref_ccf(file, time_type):
    print(f'Loading {file} ...')
    file_name = file.split('/')[-1].split('.zarr')[0]
    st1, st2 = file_name.split('_')

    z = zarr.open(f'gs://{file}')

    date_lst = z['date'][:]
    ccf_ref = z['reference'][:]
    ccf_all = z['data'][:]

    date_lst = [str(date) for date in date_lst]
    if time_type == "DAYS":
        date_lst = [f"{date[:4]}-{date[4:6]}-{date[6:]}" for date in date_lst]
    elif time_type == "HOURS":
        date_lst = [f"{date[:4]}-{date[4:6]}-{date[6:8]}T{date[8:10]}:{date[10:12]}" for date in date_lst]

    return st1, st2, date_lst, ccf_all, ccf_ref

def process_file(file, config, component):
    filterid_mwcs = config["filterid_mwcs"]
    mov_stack = config["mov_stack"]
    sampling_rate = config["sampling_rate"]
    f_low = config["f_low"]
    f_high = config["f_high"]
    maxlag = config["maxlag"]
    mwcs_wlen = config["mwcs_wlen"]
    mwcs_step = config["mwcs_step"]
    folder_mwcs = config["folder_mwcs"]
    time_type = config["time_type"]

    fs_local = gcsfs.GCSFileSystem()
    st1, st2, date_lst, ccf_all_stacked, ccf_ref = load_stacked_ref_ccf(file, time_type)
    
    # === mwcs ===
    ref_name = f'{st1}_{st2}'
    print(f"[PID {os.getpid()}] Processing pair {ref_name} with "
          f"f_low={f_low}, f_high={f_high}, maxlag={maxlag}, "
          f"mwcs_wlen={mwcs_wlen}, mwcs_step={mwcs_step}")

    outfolder = os.path.join(
            folder_mwcs, 
            'MWCS', 
            "%02i" % filterid_mwcs, 
            f"%03i_{time_type}" % mov_stack, 
            component, 
            ref_name)

    if not fs_local.exists(outfolder):
        fs_local.makedirs(outfolder, exist_ok=True)

    def process_one_date(idx_date):
        i, date = idx_date
        output = mwcs(
            current=ccf_all_stacked[i], 
            reference=ccf_ref, 
            freqmin=f_low, 
            freqmax=f_high, 
            df=sampling_rate, 
            tmin=-maxlag, 
            window_length=mwcs_wlen, 
            step=mwcs_step
            )

        filename = os.path.join(outfolder, "%s.txt" % str(date))
        
        with fs_local.open(filename, 'wt') as gcs_file:
            np.savetxt(gcs_file, output, fmt='%.16g')

        print(f"[PID {os.getpid()}] ✅ Saved: gs://{filename}")
        return date
    
    inner_workers = min(8, len(date_lst))
    print(f"[PID {os.getpid()}]  Parallelizing {len(date_lst)} dates with {inner_workers} threads")

    with ThreadPoolExecutor(max_workers=inner_workers) as executor:
        futures = {
            executor.submit(process_one_date, (i, date)): (i, date)
            for i, date in enumerate(date_lst)
        }

        for fut in as_completed(futures):
            (i, date) = futures[fut]
            try:
                _ = fut.result()
            except Exception as e:
                print(f"[PID {os.getpid()}] ❌ Error on {ref_name} date {date}: {e}")

    return ref_name

def main(node_rank, num_nodes):
    cfg = importlib.import_module(f"configs.{args.project}")
    component = cfg.component_mwcs
    path_gs = cfg.path_gs
    time_type = cfg.time_type
    # half_window = cfg.half_window

    config = {"filterid_mwcs": cfg.filterid_mwcs,
              "mov_stack": cfg.mov_stack,
              "sampling_rate": cfg.sampling_rate,
              "f_low": cfg.f_low_dic[cfg.filterid_mwcs],
              "f_high": cfg.f_high_dic[cfg.filterid_mwcs],
              "maxlag": cfg.maxlag,
              "mwcs_wlen": cfg.mwcs_wlen,
              "mwcs_step": cfg.mwcs_step,
              "folder_mwcs": cfg.folder_mwcs,
              "time_type": time_type}

    fs = gcsfs.GCSFileSystem()
    fs.invalidate_cache(path_gs)
    file_lst = fs.ls(f'{cfg.folder_stacks}{cfg.filterid_mwcs:02d}/pm{cfg.half_window:03d}_{time_type}/{component}/')
    file_lst.sort()

    files_to_process = file_lst[node_rank::num_nodes]
    print(f"Total files to process in parallel: {len(files_to_process)}/{len(file_lst)}")

    max_workers = os.cpu_count() or 1
    print(f"Using {max_workers} processes (outer level)")

    pair_lst_mwcs = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, f, config, component): f for f in files_to_process
        }

        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                pair_name = future.result()
                pair_lst_mwcs.append(pair_name)
                print(f"✅ Finished pair {pair_name} from file {file}")
            except Exception as e:
                print(f"❌ Error processing file {file}: {e}")

    print("All done.")
    print(f"Total processed pairs: {len(pair_lst_mwcs)}")


if __name__ == "__main__":
    node_rank = args.node_rank
    num_nodes = args.num_nodes
    
    main(node_rank, num_nodes)