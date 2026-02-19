from datetime import datetime, timedelta
from pathlib import Path
import fsspec

'''Generate files list from the paths on Monterey Bay DAS server, with the concate_length and batch_size specified. 
   The generated files list will be used for downchannel and downsampling the DAS data.'''

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Generate downsampled list", add_help=add_help)
    parser.add_argument(
        "--files_lst",
        default="mbdas_h5_file_list_until_20260203.txt",
        type=str,
        help="Path to the DAS files list",
    )
    parser.add_argument("--protocol", type=str, default="gs")
    parser.add_argument("--bucket_das", type=str, default="gs://cctorch/ambient_noise_das")
    parser.add_argument("--save_path", default="mbdas_h5_downsample_list", type=str, help="save path for downsampled list")
    parser.add_argument("--concate_length", default=4, type=int, help="length of concatenated files")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size for in each downsampled file")
    parser.add_argument("--time_start", default="2023-01-01T00:00", type=str, help="start time for downsampling")
    parser.add_argument("--time_end", default="2026-02-03T00:00", type=str, help="end time for downsampling")
    return parser

def get_files_dic(das_files_lst):
    f = open(das_files_lst, "r")
    das_files = f.read().splitlines()
    f.close()

    das_files_dic = {}
    for file in das_files:
        date_time_str = file.split('/')[-1].split('_')[-1].split('.')[0]
        date_time = datetime.strptime(date_time_str, '%Y-%m-%dT%H%M%SZ')
        if date_time in das_files_dic:
            print('duplicate time found:', date_time)
        else:
            das_files_dic[date_time.strftime('%Y_%j_%H_%M')] = file

    return das_files_dic

def get_datetime_lst(time_start, time_end):
    datetime_start = datetime.strptime(time_start, '%Y-%m-%dT%H:%M')
    datetime_end = datetime.strptime(time_end, '%Y-%m-%dT%H:%M')
    total_minutes = int((datetime_end - datetime_start).total_seconds() / 60)
    datetime_lst = [(datetime_start + timedelta(minutes=i)).strftime('%Y_%j_%H_%M') for i in range(total_minutes)]
    return datetime_lst

def get_hdf5_file_by_datetime(hdf5_files_dic, datetime_lst, concate_length=4, batch_size=3, save_path='.', config=None):
    protocal = config["protocol"]
    bucket = config["bucket"]
    save_path = config["save_path"]

    fs = fsspec.filesystem(protocal, token='google_default')

    lines = ['file_name\n']
    files_lst = []
    temp_off = False
    for i, datetime_temp in enumerate(datetime_lst):
        print(f"Processing datetime: {datetime_temp}, len(files_lst)={len(files_lst)}, len(lines)={len(lines)}")
        try:
            if i%(batch_size*concate_length)==0:
                temp_off = False
                date_time_start = datetime_temp
            if temp_off:
                continue
            file = hdf5_files_dic[datetime_temp]
            files_lst.append(f"{file}")
            if len(files_lst) == concate_length:
                files_lst_str = '|'.join(files_lst)
                lines.append(f"{files_lst_str}\n")
                files_lst = []
            if len(lines) == batch_size + 1:
                print(f"Writing to file: {save_path}/mbdas_h5_down_{date_time_start}.txt")
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                file_save = f"mbdas_h5_down_{date_time_start}.txt"
                with open(file_save, "w") as f:
                    for s in lines:
                        f.write(s)
                fs.put(file_save, f"{bucket}/{save_path}/{file_save}")
                lines = ['file_name\n']
        except:
            print(f"File not found for datetime: {datetime_temp}")
            lines = ['file_name\n']
            files_lst = []
            temp_off = True



if __name__ == "__main__":
    args = get_args_parser().parse_args()

    config = {"protocol": args.protocol,
              "bucket": args.bucket_das,
              "save_path": args.save_path}

    das_files_dic = get_files_dic(args.files_lst)
    file_datetime_lst = get_datetime_lst(args.time_start, args.time_end)
    get_hdf5_file_by_datetime(das_files_dic, file_datetime_lst, concate_length=args.concate_length, batch_size=args.batch_size, save_path=args.save_path, config=config)