import os
import fsspec
from datetime import datetime

'''Generate files list from the paths on GCloud, based on the files after doing down_channel. 
   The generated files list will be used for downsampling the DAS data.'''

# load_path = 'gs://cctorch/ambient_noise_das/waveforms_das/MBARI/'
# save_path = 'gs://cctorch/ambient_noise_das/das_preprocess/mbdas_h5_down_channel_sample_list/'

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Generate downsampled list", add_help=add_help)
    parser.add_argument(
        "--files_lst",
        default="DAS_h5_file_list.txt",
        type=str,
        help="Path to the DAS files list",
    )
    parser.add_argument("--protocol", type=str, default="gs")
    parser.add_argument("--bucket_das", type=str, default="gs://cctorch/ambient_noise_das")
    parser.add_argument("--load_path", default="waveforms_das/MBARI", type=str, help="load path for downsampled list")
    parser.add_argument("--save_path", default="das_preprocess/mbdas_h5_down_channel_sample_list", type=str, help="save path for downsampled list")
    parser.add_argument("--year", default=2024, type=int, help="year for downsampling")
    parser.add_argument("--jday_start", default=1, type=int, help="start julian day for downsampling")
    parser.add_argument("--jday_end", default=1, type=int, help="end julian day for downsampling")
    return parser

def regroup_files(downchannel_file_lst, date_str):
    hr_lst = [f"{i:02d}" for i in range(0, 24)]
    new_lst = []
    for hr in hr_lst:
        temp_lst = []
        for downchannel_file in downchannel_file_lst:
            if f"{date_str}T{hr}" in downchannel_file:
                temp_lst.append(downchannel_file)
        new_lst.append("|".join(temp_lst))
    return new_lst

def gen_down_channel_sample_lst(year=2024, jday_start=1, jday_end=366, config=None):
    protocol = config["protocol"]
    bucket = config["bucket"]
    load_path = config["load_path"]
    save_path = config["save_path"]

    fs = fsspec.filesystem(protocol, token='google_default')

    jday_lst = [f"{i:03d}" for i in range(jday_start, jday_end+1)]

    for jday in jday_lst:
        date = datetime.strptime(f"{year}-{jday}", "%Y-%j")
        date_str = date.strftime("%Y-%m-%d")
        print(f"[GCloud] Processing jday = {jday}, date = {date_str}")

        try:
            downchannel_file_lst = fs.ls(f'{bucket}/{load_path}/{year}/{jday}/downchannel')
            print(f"[GCloud] jday = {jday}, number of downchannel files = {len(downchannel_file_lst)}")
            downchannel_file_lst_new = regroup_files(downchannel_file_lst, date_str)
            print(f"[GCloud] jday = {jday}, number of regrouped downchannel files = {len(downchannel_file_lst_new)}")

            for i, downchannel_file_lst_merge in enumerate(downchannel_file_lst_new):
                lines = ['file_name\n'] + [downchannel_file_lst_merge]
                file_save = f'mbdas_h5_down_{year}_{jday}_{i:02d}_00.txt'
                with open(file_save, 'w') as f:
                    for line in lines:
                        f.write(line)
                fs.put(file_save, f'{bucket}/{save_path}/{file_save}')
                print(f"{file_save} -> {bucket}/{save_path}/{file_save}")

                try:
                    os.remove(file_save)
                except Exception as e:
                    print(f"Error removing local file {file_save}:\n{e}")
        except Exception as e:
            print(f"Error processing jday = {jday}:\n{e}")
            continue

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    year = args.year
    jday_start = args.jday_start
    jday_end = args.jday_end

    config = {"protocol": args.protocol,
              "bucket": args.bucket_das,
              "load_path": args.load_path,
              "save_path": args.save_path}

    gen_down_channel_sample_lst(year=year, jday_start=jday_start, jday_end=jday_end, config=config)