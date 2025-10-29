import datetime
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import fsspec
import numpy as np
import obspy
import pandas as pd
from args import parse_args
from tqdm import tqdm

from utils import get_response_paz, check_and_phase_shift

def downsample_mseed(fname, highpass_filter=False, sampling_rate=20, root_path="./", config=None):
    """
    root_path: mapping root_path/xxx to bucket/xxx
    """

    protocol = config["protocol"]
    token_file = config["token_file"]
    bucket = config["bucket"]
    fs = fsspec.filesystem(protocol, token=token_file)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    waveforms_dir = "waveforms"
    if not os.path.exists(f"{root_path}/{waveforms_dir}"):
        os.makedirs(f"{root_path}/{waveforms_dir}")

    fname_local = {}
    try:
        streams = obspy.Stream()
        datalesspz_lst = []
        for tmp in fname.split("|"):
            try:
                datalesspz = get_response_paz(tmp)
            except:
                datalesspz = None
            with fsspec.open(tmp, "rb", anon=True) as fp:
                if tmp.endswith(".sac"):
                    meta = obspy.read(fp, format="SAC")
                else:
                    meta = obspy.read(fp, format="MSEED")
                for i in range(len(meta)):
                    datalesspz_lst.append(datalesspz)
                streams += meta
            fname_local[meta[0].stats.component] = tmp.split("/")[-1]
            # stream += obspy.read(tmp)
        # stream = stream.merge(fill_value="latest")
        # stream.detrend("demean")
        for i, trace in enumerate(streams):
            trace_temp = check_and_phase_shift(trace, 0)
            trace_temp.detrend("demean")
            trace_temp.detrend("linear")
            trace_temp.taper(max_percentage=None, max_length=0)

            if len(trace_temp.data) < 10:
                continue
            trace_temp.filter("highpass", freq=0.001, zerophase=True, corners=4)
            
            if trace_temp.stats.sampling_rate != sampling_rate:
                # logging.warning(f"Resampling {trace_temp.id} from {trace_temp.stats.sampling_rate} to {sampling_rate} Hz")
                try:
                    # print(f"Resample from {trace.stats.sampling_rate} to {sampling_rate} Hz")
                    trace_temp.filter("lowpass", freq=9.8, zerophase=True, corners=8)
                    trace_temp.data = np.array(trace_temp.data)
                    trace_temp.interpolate(method="lanczos", sampling_rate=sampling_rate, a=1.0)
                except Exception as e:
                    print(f"Error resampling {trace.id}:\n{e}")

            if datalesspz_lst[i]:
                trace_temp.simulate(paz_remove=datalesspz_lst[i],
                        remove_sensitivity=True,
                        pre_filt=[0.001, 0.002, 9.8, 10],
                        paz_simulate=None, )
            else:
                print(f"Warning: No response found for {trace_temp.id}, skipping instrument removal")
                trace_temp.filter('bandpass', freqmin=0.002, freqmax=9.8, corners=2, zerophase=True)
            streams[i] = trace_temp

        streams.merge(fill_value=None)

        # ## FIXME: HARDCODE for California
        # year, jday = None, None
        # if tmp.startswith("s3://ncedc-pds"):
        #     year, jday = tmp.split("/")[-1].split(".")[-2:]
        #     year, jday = int(year), int(jday)
        #     begin_time = obspy.UTCDateTime(year=year, julday=jday)
        #     end_time = begin_time + 86400  ## 1 day
        #     stream = stream.trim(begin_time, end_time, pad=True, fill_value=0, nearest_sample=True)
        # elif tmp.startswith("s3://scedc-pds"):
        #     year_jday = tmp.split("/")[-1].rstrip(".ms")[-7:]
        #     year, jday = int(year_jday[:4]), int(year_jday[4:])
        #     begin_time = obspy.UTCDateTime(year=year, julday=jday)
        #     end_time = begin_time + 86400  ## 1 day
        #     stream = stream.trim(begin_time, end_time, pad=True, fill_value=0, nearest_sample=True)
    except Exception as e:
        print(f"Error reading {fname}:\n{e}")
        return None

    if len(streams) == 0:
        return None

    for stream in streams:
        stream.data = stream.data.astype(np.float32)
        starttime = stream.stats.starttime
        endtime = stream.stats.endtime
        midtime = starttime + (endtime - starttime) / 2
        year = midtime.year
        jday = midtime.julday
        # if year is None:
        #     year = starttime.year
        # if jday is None:
        #     jday = starttime.julday
        network = stream.stats.network
        station = stream.stats.station
        location = stream.stats.location
        channel = stream.stats.channel
        fname = f"{network}.{station}.{location}.{channel}.mseed"
        mseed_dir = f"{waveforms_dir}/{network}/{year:04d}/{jday:03d}"
        if not os.path.exists(mseed_dir):
            os.makedirs(mseed_dir, exist_ok=True)
        stream_temp = stream.split()
        stream_temp.write(f"{root_path}/{mseed_dir}/{fname}", format="MSEED")
        if protocol != "file":
            # print(f"Uploading {fname} to {bucket}/{mseed_dir}/{fname}")
            fs.put(f"{root_path}/{mseed_dir}/{fname}", f"{bucket}/{mseed_dir}/{fname}")
            os.remove(f"{root_path}/{mseed_dir}/{fname}")


if __name__ == "__main__":

    args = parse_args()
    year = f"{args.year:04d}"
    jday = f"{args.jday:03d}"
    root_path = args.root_path
    protocol = args.protocol
    token_file = args.token_file
    bucket = args.bucket

    fs = fsspec.filesystem(protocol, token=token_file)
    with fs.open(f"{bucket}/mseed_list/mseeds1_{year}_{jday}.txt", "r") as f:
        mseeds = pd.read_csv(f)["file_name"].tolist()

    config = {
        "protocol": protocol,
        "token_file": token_file,
        "bucket": bucket,
    }

    num_workers = os.cpu_count()
    print(f"Processing {len(mseeds)} files using {num_workers} workers")

    # with ThreadPoolExecutor(max_workers=num_workers) as executor:
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for mseed in mseeds:
            future = executor.submit(
                downsample_mseed,
                mseed,
                highpass_filter=False,
                sampling_rate=20,
                root_path=root_path,
                config=config,
            )
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downsampling"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {e}")
