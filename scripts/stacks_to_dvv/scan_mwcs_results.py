import gcsfs
import importlib

from args import parse_args
args = parse_args()

project = args.project
bucket = args.bucket_das
path_gs = bucket[:5]
folder_gs = bucket[5:]

cfg = importlib.import_module(f"configs.{project}")
time_type = cfg.time_type
component = cfg.component_mwcs

fs = gcsfs.GCSFileSystem()
fs.invalidate_cache(path_gs)

files = fs.ls(f'{folder_gs}/mwcs_{project}/MWCS/01/001_{time_type}/{component}')

file_save = f'mwcs_paths_{project}_{component.lower()}.txt'
save_path = 'das_preprocess/dtt_info'
with open(file_save, 'w') as f:
    for line in files:
        f.write(line)
fs.put(file_save, f'{bucket}/{save_path}/{file_save}')
print(f"{file_save} -> {bucket}/{save_path}/{file_save}")