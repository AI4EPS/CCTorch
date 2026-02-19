import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import fsspec
import pandas as pd
import sky
from args import parse_args
from tqdm import tqdm

import sys

###### Hardcoded #######
token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
###### Hardcoded #######

args = parse_args()
NUM_NODES = args.num_nodes
YEAR = args.year
JDAY_START = args.jday_start
JDAY_END = args.jday_end
HR = args.hr
HR_END = args.hr_end
MINUTE = args.minute
FILE_MARK = args.file_mark

task = sky.Task(
    name=f"run-downchannel_das",
    setup="""
echo "Begin setup."                                                           
echo export WANDB_API_KEY=$WANDB_API_KEY >> ~/.bashrc
pip install h5py tqdm wandb pandas scipy scikit-learn numpy==1.26.4
pip install fsspec gcsfs s3fs                                                   
pip install obspy pyproj
pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cpu

""",
    run="""
num_nodes=`echo "$SKYPILOT_NODE_IPS" | wc -l`
master_addr=`echo "$SKYPILOT_NODE_IPS" | head -n1`
if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
    ls -al /opt
    ls -al /data
    ls -al ./
fi
echo "Running downchannel on (node_rank, num_node) = ($NODE_RANK, $NUM_NODES)"
echo "YEAR=$YEAR, JDAY_START=$JDAY_START, JDAY_END=$JDAY_END, HR=$HR, HR_END=$HR_END, MINUTE=$MINUTE"
python run_downchannel_das.py --year $YEAR --node_rank $NODE_RANK --num_nodes $NUM_NODES --jday_start $JDAY_START --jday_end $JDAY_END --hr $HR --hr_end $HR_END --minute $MINUTE
""",
    workdir=".",
    num_nodes=1,
    envs={"YEAR": YEAR, "NUM_NODES": NUM_NODES, "NODE_RANK": 0, "JDAY_START": JDAY_START, "JDAY_END": JDAY_END, 
          "HR": HR, "HR_END": HR_END, "MINUTE": MINUTE, "FILE_MARK": FILE_MARK},
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
        # instance_type="n2-standard-16",
        # instance_type="n2-highmem-2",
        instance_type="t2d-standard-1",
        cpus=1,
        # cpus=16,
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
# # sky.launch(task, cluster_name="downchannel")
# sky.exec(task, cluster_name="downchannel")
# raise

job_idx = 1
requests_ids = []
for NODE_RANK in range(NUM_NODES):
    # for NODE_RANK in range(30):

    task.update_envs({"NODE_RANK": NODE_RANK})
    cluster_name = f"downchannel-{FILE_MARK}-{NODE_RANK:03d}"

    # requests_ids.append(sky.jobs.launch(task, name=f"{cluster_name}"))
    requests_ids.append(sky.launch(task, cluster_name=f"{cluster_name}"))

    print(f"Running downchannel DAS on (rank={NODE_RANK}, num_node={NUM_NODES}) of {cluster_name}")

    job_idx += 1

# for request_id in requests_ids:
#     print(sky.get(request_id))

# (Optional) stream the logs from the task to the console.
job_id, handle = sky.stream_and_get(requests_ids[0])
cluster_name = handle.get_cluster_name()
returncode = sky.tail_logs(cluster_name, job_id, follow=True)

sys.exit(returncode)