project = "la_mao_2025"
filterid_list = [1]
time_type = "DAYS"

# --- for stacking ---
station_pair_list = f"configs/station_pair_list/station_pair_{project}.txt"

f_low_dic = {1:0.2}
f_high_dic = {1:0.8}

ref_start = "2000-01-01"
ref_end   = "2025-12-31"

ccf_start = "2000-01-01"
ccf_end   = "2025-12-31"

sampling_rate = 20
half_window_days = 10

component_list = ["EE", "NN", "ZZ"]

folder_stacks = f'cctorch/ambient_noise/stacks_{project}/'

# --- for mwcs ---
filterid_mwcs = 1
mov_stack = 30
pre_f_low = 0.08
pre_f_high = 8.0

maxlag = 300
mwcs_wlen = 8
mwcs_step = 1

component_mwcs = "ZZ"

path_gs = "gs://"
folder_mwcs = f'cctorch/ambient_noise/mwcs_{project}/'

# --- for dtt ---
mwcs_paths = f'configs/dtt_info/mwcs_paths_{project}_{component_mwcs.lower()}.txt'

dynamic_dtt_minlag = False
file_dtt_minlag = 'configs/dtt_info/dtt_minlag.txt'

filterid_dtt = 1

dtt_minlag_default = 18
dtt_v = 1
dtt_width = 42
dtt_sides = 'both'

dtt_mincoh = 0.5
dtt_maxerr = 0.1
dtt_maxdt = 0.5

mov_stacks = [mov_stack]

folder_dtt = f'cctorch/ambient_noise/dtt_{project}/'

# --- for dvv ---
dtt_paths = f'configs/dtt_info/dtt_paths_{project}_{component_mwcs.lower()}.txt'
folder_dvv = f'cctorch/ambient_noise/dvv_{project}/'