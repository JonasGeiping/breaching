#!/usr/bin/env python3

r"""Launch file to dispatch jobs onto the Omni SLURM computation cluster.

Messing up and launching a large number of jobs that fail immediately can slow down the cluster!
Launch debug options and small tests until everything runs as it is supposed to be.

-----------------------------------------------------------------------------------------------------------------------
USAGE: python omni.py file_list --arguments
USAGE: python omni.py cmd_string --arguments

- Required: file_list : A file containing python calls with a single call per line.
The file may contain spaces, empty lines and comments (#). These will be scrubbed automatically.
Example file_list.sh:

'''
python something.py --someargument # this is the first job

# this is the second job:
python something.py --someotherargument
'''

OR

- Required: cmd_string : A string for a single command


- Necessary Options:
--conda /path/to/your/env - Set the path to your anaconda installation here! [Or change the default to your installation]

- Optional but important:
--email                   - Set a custom email to receive cancellation emails if not username@uni-siegen.de
--name                    - Set a custom name for the job  that will display in squeue.

- Quality of service options:
--nodes, --gpus, --mem,     - These options work exactly as in the usual srun commands. NOTE THAT ALL OPTIONS ARE PER JOB.
--timelimit                 Use "spartition" and "sinfo" on the login shell to see your options.
                            16 CPU cores are automatically allocated per GPU.

--throttling              - Use this option to reduce the number of jobs that will be launched simultaneously.



If you do not want to use anaconda, hard-code or otherwise modify the lines

source {"/".join(args.conda.split("/")[:-2])}/etc/profile.d/conda.sh
conda activate {args.conda}

in SBATCH_PROTOTYPE [defined around l199 in this file] to initialize your personal choice of environment.

------- BASIC SLURM commands -----------------------------------------------------------------------------------------
- Show the cluster status and queue:
    squeue
- Cancel a job/job_array:
    scancel job_id
- Show cluster information:
    sinfo
    spartittion
- Show information about your recents jobs
    sacct

- Pause a job in the queue that has not started yet:
    scontrol hold job_id

- Use "ssh nodename" to log into a node to check nvidia-smi or top

-----------------------------------------------------------------------------------------------------------------------
------- OTHER helpful SLURM commands ----------------------------------------------------------------------------------
- Top GPU usage over some time period:
    sreport -tminper user Top --tres="gres/gpu" start=2021-03-01 end=now TopCount=100
- Full stats over some time period:
    sreport -T gres/gpu,cpu   cluster accountutilizationbyuser start=03/01/21T00:00:00 end=now -t hours
- SShare coefficients personal / for all users
    sshare / sshare -a
- Current priority setup:
    scontrol show config | grep Priority
- Update job to start after another job:
    scontrol update job=[255795] dependency=255776_40
- Detailed information about job cancellations and job terminations:
    sacct -a -o JobID,JobName,Partition,Account,State,ExitCode,User -S today
- Count total number of submitted jobs for some user:
    sacct -u jg371845 --start=2021-03-01 --end=now | grep batch > jobcount &&  wc -l jobcount
- Show quality of service information:
    sacctmgr show qos format=name,MaxJobsPU,MaxTRES%35
- Who is who?
    getent passwd jg371845 | cut -d : -f 5       (not actually a SLURM command)
-----------------------------------------------------------------------------------------------------------------------
PyTorch multi-node jobs:
- Use env:// initialization
- Manually set the node rank to int(os.environ["SLURM_NODEID"]) within your code, not within the sbatch file
- Supply world_size=args.nodes as argument to your code (or also just query from SLURM_NTASKS)
-----------------------------------------------------------------------------------------------------------------------
JONAS GEIPING, Computer Vision Group 2021. Don't blame me though :>

"""

import argparse
import os
import subprocess
import time
import warnings
import getpass
import random

parser = argparse.ArgumentParser(description="Dispatch a list of python jobs from a given file to the CML cluster")

# Central:
parser.add_argument("file")

parser.add_argument("--conda", default="/home/jg371845/miniconda3/envs/dl", type=str, help="Path to conda env")
parser.add_argument("--email", default="jonas.geiping@uni-siegen.de", type=str, help="Your email for status messages.")

parser.add_argument("--name", default=None, type=str, help="Name that will be displayed in squeue. Default: file name")
parser.add_argument("--gpus", default="1", type=int, help="Requested GPUs PER job")
parser.add_argument("--nodes", default="1", type=int, help="Requested number of nodes.")
parser.add_argument("--mem", default="60", type=int, help="Requested memory PER job")
parser.add_argument("--timelimit", default=24, type=int, help="Requested hour limit PER job")
parser.add_argument("--throttling", default=None, type=int, help="Launch only this many jobs concurrently")
parser.add_argument(
    "--exclude", default=None, type=str, help="Exclude malfunctioning nodes. Has to be a string like gpu-node009"
)


args = parser.parse_args()

if os.path.isfile(args.file):
    job_list = True
else:
    job_list = False

# Parse and validate input:
if args.name is None:
    dispatch_name = args.file
else:
    dispatch_name = args.name

args.conda = args.conda.rstrip("/")

# Usage warnings:
if args.mem > 240:
    raise ValueError("Maximal node memory exceeded.")
if args.gpus > 4:
    raise ValueError("Maximal node GPU number exceeded.")
if args.nodes > 2:
    raise ValueError("Maximal number of nodes exceeded.")
if args.mem / args.gpus > 60:
    warnings.warn(
        "You are oversubscribing to memory. " "This might leave some GPUs idle as total node memory is consumed."
    )
if args.timelimit > 24:
    warnings.warn("GPU partition only allows for 24 hours. Timelimit request has been reduced to 24.")
    args.timelimit = 24

# 1) Strip the file_list of comments and blank lines
if job_list:
    content = open(args.file, "r").readlines()
else:
    content = args.file

jobs = [c.strip().split("#", 1)[0] for c in content if "python" in c and c[0] != "#"]
print(f"Detected {len(jobs)} jobs.")

if len(jobs) < 1:
    raise ValueError("No valid jobs detected.")

if job_list:
    # Write the clean file list
    authkey = random.randint(10 ** 5, 10 ** 6 - 1)
    with open(f".cml_job_list_{authkey}.temp.sh", "w") as file:
        file.writelines(chr(10).join(job for job in jobs))
        file.write("\n")


username = getpass.getuser()
# 3) Prepare environment
if not os.path.exists("log"):
    os.makedirs("log")

# 4) Construct the sbatch launch file

SBATCH_PROTOTYPE = f"""#!/bin/bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name={''.join(e for e in dispatch_name if e.isalnum())}
#SBATCH --array={f"1-{len(jobs)}" if args.throttling is None else f"1-{len(jobs)}%{args.throttling}"}
{f"#SBATCH --exclude={args.exclude}" if args.exclude is not None else ''}
{f"#SBATCH --nodes={args.nodes}" if args.nodes > 1 else ''}
{f"#SBATCH --ntasks={args.nodes}" if args.nodes > 1 else ''}
{f"#SBATCH --ntasks-per-node=1" if args.nodes > 1 else ''}
{f"#SBATCH --wait-all-nodes=1" if args.nodes > 1 else ''}
#SBATCH --output=log/%x_%A_%a.log
#SBATCH --error=log/%x_%A_%a.log
#SBATCH --time={args.timelimit}:00:00
#SBATCH --account=uni-siegen
#SBATCH --qos=normal
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --cpus-per-task={min(args.gpus * 16, 64)}
#SBATCH --partition=gpu
#SBATCH --mem={args.mem}gb
#SBATCH --mail-user={args.email if args.email is not None else username + "@uni-siegen.de"}
#SBATCH --mail-type=FAIL,ARRAY_TASKS

source {"/".join(args.conda.split("/")[:-2])}/etc/profile.d/conda.sh
conda activate {args.conda}

export MASTER_PORT=$(shuf -i 2000-65000 -n 1)
export MASTER_ADDR=`/bin/hostname -s`

{f"srun $(head -n $((${{SLURM_ARRAY_TASK_ID}})) .cml_job_list_{authkey}.temp.sh | tail -n 1)" if job_list else jobs[0]}

"""

# Write launch commands to file
with open(f".cml_launch_{authkey}.temp.sh", "w") as file:
    file.write(SBATCH_PROTOTYPE)


# 5) Print launch information

print("Launch prototype is ...")
print("---------------")
print(SBATCH_PROTOTYPE)
print("---------------")
print(chr(10).join("srun " + job for job in jobs))
print(f"Preparing {len(jobs)} jobs as user {username} for launch in 10 seconds...")
print("Terminate if necessary ...")
for _ in range(10):
    time.sleep(1)

# 6) Launch

# Execute file with sbatch
subprocess.run(["sbatch", f".cml_launch_{authkey}.temp.sh"])
print("Subprocess launched ...")
time.sleep(3)
subprocess.run("squeue")
