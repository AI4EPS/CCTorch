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

import sys
current_dir = os.path.dirname(__file__)
cctorch_path = os.path.abspath(os.path.join(current_dir, '../../cctorch'))
sys.path.insert(0, cctorch_path)

from utils import get_response_paz, check_and_phase_shift

# from obspy import read_inventory
# from obspy.core import UTCDateTime
# import scipy.fft as sf
# from scipy.fft import next_fast_len

# def get_instrument_resp(network, station):
#     fs_s3 = fsspec.filesystem('s3')
#     file_instru_resp = f"s3://ncedc-pds/FDSNstationXML/{network}/{network}.{station}.xml"
#     try:
#         with fs_s3.open(file_instru_resp, "rb") as f:
#             return read_inventory(f)
#     except:
#         if network in ["CI", "ZY"]:
#             network_folder = "CI"
#         else:
#             network_folder = "unauthoritative-XML"
#         file_instru_resp = f"s3://scedc-pds/FDSNstationXML/{network_folder}/{network}_{station}.xml"
#         with fs_s3.open(file_instru_resp, "rb") as f:
#             return read_inventory(f)

# def get_nc_info(fname):
#     file = fname.split('/')[-1]
#     network = file.split(".")[1]
#     station = file.split(".")[0]
#     channel = file.split(".")[2]
#     location = file.split(".")[3]
#     year = file.split(".")[-2]
#     jday = file.split(".")[-1]
#     date_target = obspy.UTCDateTime(year=int(year), julday=int(jday))
#     channel_id = f"{network}.{station}.{location}.{channel}"
#     return network, station, channel_id, date_target

# def get_sc_info(fname):
#     file = fname.split('/')[-1]
#     network = file[:2]
#     station = file[2:7].strip("_")
#     channel = file[7:10]
#     year, jday = int(file[-10:-6]), int(file[-6:-3])
#     date_target = obspy.UTCDateTime(year=year, julday=jday)
#     channel_id = f"{network}.{station}..{channel}"
#     return network, station, channel_id, date_target

# def get_instrument_response(fname):
#     print(f"Finding instrument response for {fname}...")
#     if fname.startswith("s3://ncedc-pds"):
#         network, station, channel_id, date_target = get_nc_info(fname)   
#     elif fname.startswith("s3://scedc-pds"):
#         network, station, channel_id, date_target = get_sc_info(fname)
#     else:
#         raise ValueError(f"Unsupported file format: {fname}")
#     return get_instrument_resp(network, station), channel_id, date_target

# def from_inv_to_pd(inv, channel_id, date_target):
#     channels = []
#     for net in inv.networks:
#         for sta in net.stations:
#             for cha in sta.channels:
#                 try:
#                     seed_id = "%s.%s.%s.%s" % (net.code, sta.code,
#                                                 cha.location_code,
#                                                 cha.code)
#                     resp = inv.get_response(seed_id, cha.start_date+10)
#                     polezerostage = resp.get_paz()
#                     totalsensitivity = resp.instrument_sensitivity
#                     pzdict = {}
#                     pzdict['poles'] = polezerostage.poles
#                     pzdict['zeros'] = polezerostage.zeros
#                     pzdict['gain'] = polezerostage.normalization_factor
#                     pzdict['sensitivity'] = totalsensitivity.value
#                     channels.append([seed_id, cha.start_date,
#                                         cha.end_date or UTCDateTime(),
#                                         pzdict, cha.latitude,
#                                         cha.longitude])
#                 except:
#                     continue
#     channels = pd.DataFrame(channels, columns=["channel_id", "start_date",
#                                                "end_date", "paz", "latitude",
#                                                "longitude"],)
#     channels = channels[channels["channel_id"] == channel_id]
#     if len(channels) > 1:
#         channels = channels[channels["start_date"] <= date_target]
#     if len(channels) > 1:
#         channels = channels[channels["end_date"] >= date_target]
#     elif len(channels) == 0:
#         raise ValueError(f"No channel found for {channel_id} at {date_target}")
#     return channels

# def get_response_paz(fname):
#     inv, channel_id, date_target = get_instrument_response(fname)
#     response = from_inv_to_pd(inv, channel_id, date_target)
#     return response["paz"].values[0]

# def check_and_phase_shift(trace, taper_length=20.0):
#     # TODO replace this hard coded taper length
#     if trace.stats.npts < 4 * taper_length*trace.stats.sampling_rate:
#         trace.data = np.zeros(trace.stats.npts)
#         return trace

#     dt = np.mod(trace.stats.starttime.datetime.microsecond*1.0e-6,
#                 trace.stats.delta)
#     if (trace.stats.delta - dt) <= np.finfo(float).eps:
#         dt = 0.
#     if dt != 0.:
#         if dt <= (trace.stats.delta / 2.):
#             dt = -dt
#         else:
#             dt = (trace.stats.delta - dt)
#         logging.debug("correcting time by %.6fs"%dt)
#         trace.detrend(type="demean")
#         trace.detrend(type="simple")
#         trace.taper(max_percentage=None, max_length=1.0)
#         n = next_fast_len(int(trace.stats.npts))
#         FFTdata = sf.fft(trace.data, n=n)
#         fftfreq = sf.fftfreq(n, d=trace.stats.delta)
#         FFTdata = FFTdata * np.exp(1j * 2. * np.pi * fftfreq * dt)
#         FFTdata = FFTdata.astype(np.complex64)
#         sf.ifft(FFTdata, n=n, overwrite_x=True)
#         trace.data = np.real(FFTdata[:len(trace.data)]).astype(np.float64)
#         trace.stats.starttime += dt
#         del FFTdata, fftfreq
#         return trace
#     else:
#         return trace

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
                logging.warning(f"Resampling {trace_temp.id} from {trace_temp.stats.sampling_rate} to {sampling_rate} Hz")
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
            print(f"Uploading {fname} to {bucket}/{mseed_dir}/{fname}")
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
