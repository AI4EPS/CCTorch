import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import fsspec
import pandas as pd
import sky
from args import parse_args
from sky import Storage, StorageMode
from tqdm import tqdm

###### Hardcoded #######
token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
with open(token_json, "r") as fp:
    token = json.load(fp)
fs = fsspec.filesystem("gs", token=token)
###### Hardcoded #######


args = parse_args()
NUM_NODES = args.num_nodes
YEAR = args.year
task = sky.Task(
    name=f"run-cctorch",
    setup="""
echo "Begin setup."                                                           
echo export WANDB_API_KEY=$WANDB_API_KEY >> ~/.bashrc
conda install python=3.11 pip
pip install h5py tqdm wandb pandas scipy scikit-learn numpy==1.26.4
pip install fsspec gcsfs s3fs zarr                                           
pip install obspy pyproj
pip install torch torchvision torchaudio
""",
    run="""
num_nodes=`echo "$SKYPILOT_NODE_IPS" | wc -l`
master_addr=`echo "$SKYPILOT_NODE_IPS" | head -n1`
if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
    ls -al /opt
    ls -al /data
    ls -al ./
fi
echo "Running CCTorch on (node_rank, num_node) = ($NODE_RANK, $NUM_NODES)"
# python run_cctorch.py --year $YEAR --node_rank $NODE_RANK --num_nodes $NUM_NODES --result_path /mnt/cctorch/ambient_noise/ccf
python run_cctorch.py --year $YEAR --node_rank $NODE_RANK --num_nodes $NUM_NODES --result_path gs://cctorch/ambient_noise/ccf
""",
    workdir=".",
    num_nodes=1,
    envs={"YEAR": YEAR, "NUM_NODES": NUM_NODES, "NODE_RANK": 0},
)

task.set_file_mounts(
    {
        "/opt/CCTorch": "../../../CCTorch",
    },
)
# task.set_storage_mounts(
#     {
#         "/mnt/cctorch": sky.Storage(source="gs://cctorch/", mode=sky.StorageMode.MOUNT_CACHED),
#         # "/mnt/cctorch": sky.Storage(source="gs://cctorch/", mode=sky.StorageMode.MOUNT),
#     },
# )

task.set_resources(
    sky.Resources(
        cloud=sky.GCP(),
        region="us-west1",  # GCP
        # region="us-west-2",  # AWS
        accelerators="V100:1",
        # cpus=8,
        disk_tier="medium",
        disk_size=200,  # GB
        memory=None,
        use_spot=True,
    ),
)

jobs = []
try:
    sky.status(refresh="AUTO")
except Exception as e:
    print(e)

# task.update_envs({"NODE_RANK": 0})
# job_id = sky.launch(task, cluster_name="cctorch", fast=True)
# # job_id = sky.exec(task, cluster_name="cctorch")
# status = sky.stream_and_get(job_id)
# # sky.tail_logs(cluster_name="cctorch8", job_id=job_id, follow=True)
# print(f"Job ID: {job_id}, status: {status}")

# raise

job_idx = 1
requests_ids = []
for NODE_RANK in range(NUM_NODES):
    # for NODE_RANK in range(30):

    task.update_envs({"NODE_RANK": NODE_RANK})
    cluster_name = f"cctorch-{NODE_RANK:03d}"

    requests_ids.append(sky.jobs.launch(task, name=f"{cluster_name}"))

    print(f"Running cctorch on (rank={NODE_RANK}, num_node={NUM_NODES}) of {cluster_name}")

    job_idx += 1

for request_id in requests_ids:
    print(sky.get(request_id))
