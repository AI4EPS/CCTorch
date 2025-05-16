# %%
import fsspec 

token_file = "./application_default_credentials.json"
fs = fsspec.filesystem("gs", token=token_file)

# %%
nc_list = fs.ls("gs://quakeflow_catalog/NC/mseed_list/")
sc_list = fs.ls("gs://quakeflow_catalog/SC/mseed_list/")

# Count the number of lines in each file
nc_counts = {}
for file_path in nc_list:
    with fs.open(file_path, 'r') as f:
        lines = f.readlines()
        nc_counts[file_path] = len(lines)
        print(f"NC file: {file_path}, lines: {len(lines)}")

sc_counts = {}
for file_path in sc_list:
    with fs.open(file_path, 'r') as f:
        lines = f.readlines()
        sc_counts[file_path] = len(lines)
        print(f"SC file: {file_path}, lines: {len(lines)}")

total_nc_lines = sum(nc_counts.values())
total_sc_lines = sum(sc_counts.values())
print(f"Total NC lines: {total_nc_lines}")
print(f"Total SC lines: {total_sc_lines}")
print(f"Total lines: {total_nc_lines + total_sc_lines}")

# %%




# %%
