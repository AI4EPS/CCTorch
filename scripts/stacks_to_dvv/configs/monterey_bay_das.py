project = "monterey_bay_das"
filterid_list = [1]
time_type = "HOURS"

# --- for stacking ---
station_pair_list = f"configs/station_pair_list/station_pair_{project}.txt"
station_merge_list = f"configs/station_merge_list/station_merge_{project}.txt"

f_low_dic = {1:0.1}
f_high_dic = {1:0.2}

# ref_start = "2023-01-01"
# ref_end   = "2024-09-30"

ref_start = "2023-01-01"
ref_end   = "2024-12-31"

ccf_start = "2023-01-01"
ccf_end   = "2024-12-31"

sampling_rate = 50
half_window = 1

component_list = ["ZZ"]

folder_stacks = f'cctorch/ambient_noise_das/stacks_{project}/'

# --- for mwcs ---
filterid_mwcs = 1
mov_stack = 1 
pre_f_low = 0.008
pre_f_high = 48.0

maxlag = 120
mwcs_wlen = 20
mwcs_step = 1

component_mwcs = "ZZ"

path_gs = "gs://"
folder_mwcs = f'cctorch/ambient_noise_das/mwcs_{project}/'

# --- for dtt ---
mwcs_paths = f'cctorch/ambient_noise_das/das_preprocess/dtt_info/mwcs_paths_{project}_{component_mwcs.lower()}.txt'

dynamic_dtt_minlag = False
file_dtt_minlag = 'cctorch/ambient_noise_das/das_preprocess/dtt_info/dtt_minlag.txt'

filterid_dtt = 1

dtt_minlag_default = 80
dtt_v = 1
dtt_width = 20
dtt_sides = 'both'

dtt_mincoh = 0.5
dtt_maxerr = 0.1
dtt_maxdt = 0.5

mov_stacks = [mov_stack]

folder_dtt = f'cctorch/ambient_noise_das/dtt_{project}/'

# --- for dvv ---
dtt_paths = f'cctorch/ambient_noise_das/das_preprocess/dtt_info/dtt_paths_{project}_{component_mwcs.lower()}.txt'
folder_dvv = f'cctorch/ambient_noise_das/dvv_{project}/'