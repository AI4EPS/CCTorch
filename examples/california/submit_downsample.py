import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import fsspec
import sky
from tqdm import tqdm
import pandas as pd
from args import parse_args

###### Hardcoded #######
token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
###### Hardcoded #######

args = parse_args()
NUM_NODES = args.num_nodes
YEAR = args.year
task = sky.Task(
    name=f"run-downsample",
    setup="""
echo "Begin setup."                                                           
echo export WANDB_API_KEY=$WANDB_API_KEY >> ~/.bashrc
pip install h5py tqdm wandb pandas scipy scikit-learn numpy==1.26.4
pip install fsspec gcsfs s3fs                                                   
pip install obspy pyproj
""",
    run="""
num_nodes=`echo "$SKYPILOT_NODE_IPS" | wc -l`
master_addr=`echo "$SKYPILOT_NODE_IPS" | head -n1`
if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
    ls -al /opt
    ls -al /data
    ls -al ./
fi
echo "Running downsample on (node_rank, num_node) = ($NODE_RANK, $NUM_NODES)"
python run_downsample.py --year $YEAR --node_rank $NODE_RANK --num_nodes $NUM_NODES
""",
    workdir=".",
    num_nodes=1,
    envs={"YEAR": YEAR, "NUM_NODES": NUM_NODES, "NODE_RANK": 0},
)

task.set_file_mounts(
    {
        "/opt/CCTorch": "../../../CCTorch",
        "~/sky_workdir/application_default_credentials.json": token_json,
    },
)

task.set_resources(
    sky.Resources(
        cloud=sky.GCP(),
        region="us-west1",  # GCP
        # region="us-west-2",  # AWS
        # accelerators="V100:1",
        cpus=8,
        disk_tier="low",
        disk_size=50,  # GB
        memory=None,
        use_spot=True,
    ),
)

jobs = []
try:
    sky.status(refresh="AUTO")
except Exception as e:
    print(e)

# task.update_envs({"NODE_RANK": 3})
# # sky.launch(task, cluster_name="downsample")
# sky.exec(task, cluster_name="downsample")
# raise

job_idx = 1
requests_ids = []
for NODE_RANK in range(NUM_NODES):
    # for NODE_RANK in range(2):
    # for NODE_RANK in range(30):

    task.update_envs({"NODE_RANK": NODE_RANK})
    cluster_name = f"downsample{NODE_RANK:03d}"

    requests_ids.append(sky.jobs.launch(task, name=f"{cluster_name}"))

    print(f"Running downsample on (rank={NODE_RANK}, num_node={NUM_NODES}) of {cluster_name}")

    job_idx += 1

for request_id in requests_ids:
    print(sky.get(request_id))
