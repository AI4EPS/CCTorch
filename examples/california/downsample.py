import fsspec
import numpy as np
import obspy
import logging
import os
from args import parse_args
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import datetime


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
        stream = obspy.Stream()
        for tmp in fname.split("|"):
            with fsspec.open(tmp, "rb", anon=True) as fp:
                if tmp.endswith(".sac"):
                    meta = obspy.read(fp, format="SAC")
                else:
                    meta = obspy.read(fp, format="MSEED")
                stream += meta
            fname_local[meta[0].stats.component] = tmp.split("/")[-1]
            # stream += obspy.read(tmp)
        stream = stream.merge(fill_value="latest")
        stream.detrend("demean")

        ## FIXME: HARDCODE for California
        year, jday = None, None
        if tmp.startswith("s3://ncedc-pds"):
            year, jday = tmp.split("/")[-1].split(".")[-2:]
            year, jday = int(year), int(jday)
            begin_time = obspy.UTCDateTime(year=year, julday=jday)
            end_time = begin_time + 86400  ## 1 day
            stream = stream.trim(begin_time, end_time, pad=True, fill_value=0, nearest_sample=True)
        elif tmp.startswith("s3://scedc-pds"):
            year_jday = tmp.split("/")[-1].rstrip(".ms")[-7:]
            year, jday = int(year_jday[:4]), int(year_jday[4:])
            begin_time = obspy.UTCDateTime(year=year, julday=jday)
            end_time = begin_time + 86400  ## 1 day
            stream = stream.trim(begin_time, end_time, pad=True, fill_value=0, nearest_sample=True)
    except Exception as e:
        print(f"Error reading {fname}:\n{e}")
        return None

    tmp_stream = obspy.Stream()
    for trace in stream:
        if len(trace.data) < 10:
            continue

        ## interpolate to 100 Hz
        if trace.stats.sampling_rate != sampling_rate:
            # logging.warning(f"Resampling {trace.id} from {trace.stats.sampling_rate} to {sampling_rate} Hz")
            try:
                trace.filter("lowpass", freq=0.45 * sampling_rate, zerophase=True, corners=8)
                trace.interpolate(method="lanczos", sampling_rate=sampling_rate, a=1.0)
                if tmp.startswith(("s3://ncedc-pds", "s3://scedc-pds")):
                    trace = trace.trim(begin_time, end_time, pad=True, fill_value=0, nearest_sample=True)
            except Exception as e:
                print(f"Error resampling {trace.id}:\n{e}")

        tmp_stream.append(trace)

    if len(tmp_stream) == 0:
        return None
    stream = tmp_stream

    begin_time = min([st.stats.starttime for st in stream])
    end_time = max([st.stats.endtime for st in stream])
    stream = stream.trim(begin_time, end_time, pad=True, fill_value=0)

    for tr in stream:
        tr.data = tr.data.astype(np.float32)
        starttime = tr.stats.starttime + datetime.timedelta(hours=12)
        year = starttime.year
        jday = starttime.julday
        # if year is None:
        #     year = starttime.year
        # if jday is None:
        #     jday = starttime.julday
        network = tr.stats.network
        station = tr.stats.station
        location = tr.stats.location
        channel = tr.stats.channel
        fname = f"{network}.{station}.{location}.{channel}.mseed"
        mseed_dir = f"{waveforms_dir}/{network}/{year:04d}/{jday:03d}"
        if not os.path.exists(mseed_dir):
            os.makedirs(mseed_dir)
        tr.write(f"{root_path}/{mseed_dir}/{fname}", format="MSEED")
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

    num_workers = 16
    print(f"Processing {len(mseeds)} files using {num_workers} workers")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
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
