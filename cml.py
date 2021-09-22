#!/usr/bin/env python3

r"""Launch file to dispatch jobs onto the CML SLURM computation cluster.

Messing up and launching a large number of jobs that fail immediately can slow down the cluster!
Launch debug options and small tests until everything runs as it is supposed to be.

To unlock 4 GPU options in high priority and other useful comments, follow the instructions at
------------------> https://wiki.umiacs.umd.edu/umiacs/index.php/CML <--------------------------------
-----------------------------------------------------------------------------------------------------------------------
USAGE: python dispatch.py file_list --arguments

- Required: file_list : A file containing python calls with a single call per line.
The file may contain spaces, empty lines and comments (#). These will be scrubbed automatically.
Example file_list.sh:

'''
python something.py --someargument # this is the first job

# this is the second job:
python something.py --someotherargument
'''

- Necessary Options:
--conda /path/to/your/env - Set the path to your anaconda installation here! [Or change the default to your installation]

- Optional but important:
--email                   - Set a custom email, if you dont check your @umiacs.umd.edu account
--name                    - Set a custom name for the job  that will display in squeue.
--min_preemptions         - Use this option to start high-priority jobs only on nodes where no scavengers are running.
                            Meaning that your high-prio jobs will wait instead of pre-empting your colleagues'
                            low-prio jobs.

- Quality of service options:
--qos, --gpus, --mem,     - These options work exactly as in the usual srun commands. NOTE THAT ALL OPTIONS ARE PER JOB.
--timelimit                 Use "show_qos" and "show_assoc" on the login shell to see your options.
                            4 CPUs are automatically allocated per GPU.

--throttling              - Use this option to reduce the number of jobs that will be launched simultaneously.



If you do not want to use anaconda, hard-code or modify the lines

source {"/".join(args.conda.split("/")[:-2])}/etc/profile.d/conda.sh
conda activate {args.conda}

in SBATCH_PROTOTYPE, replacing them with your personal choice of environment.

------- BASIC SLURM commands -----------------------------------------------------------------------------------------
- Show the cluster status and queue:
    squeue
- Cancel a job/job_array:
    scancel job_id
- Show cluster information:
    sinfo
- Show information about your recents jobs
    sacct
- Interactive session:
    High Priority: srun --pty --gres=gpu:1 --mem=16G --account=cml --partition=dpart --qos=default \
                        --cpus-per-task=4 --time=04:00:00 bash
    Low Priorty:   srun --pty --gres=gpu:1 --mem=16G --account=scavenger --partition=scavenger --qos=scavenger \
                        --cpus-per-task=4 --time=04:00:00 bash

- Pause a job in the queue that has not started yet:
    scontrol hold job_id

-----------------------------------------------------------------------------------------------------------------------
------- OTHER helpful SLURM commands ----------------------------------------------------------------------------------
- Top GPU usage over some time period:
    sreport -tminper user Top --tres="gres/gpu" start=2019-11-01 end=now TopCount=100
- Full scavenger stats over some time period:
    sreport -T gres/gpu,cpu   cluster accountutilizationbyuser start=11/01/19T00:00:00 end=now -t hours account=scavenger
- SShare coefficients personal / for all users
    sshare / sshare -a
- Current priority setup:
    scontrol show config | grep Priority
- Detailed information about job cancellations and job terminations:
    sacct -a -o JobID,JobName,Partition,Account,State,ExitCode,User -S today
- Count total number of submitted jobs for some user:
    sacct -u tomg --start=2019-09-01 --end=now | grep batch > jobcount &&  wc -l jobcount
- Show quality of service information:
    sacctmgr show qos format=name,MaxJobsPU,MaxTRES%35,MaxWallDurationPerJob
- Show node information (setup, CPUs, GPUs, OS):
    scontrol show node
-Print detailed information about (everyone's) recent jobs:
    sacct --allusers --format=User,JobID,QOS,start,end,time,ReqMem,ncpus,nodelist,AllocTRES%35

-----------------------------------------------------------------------------------------------------------------------

"""

import argparse
import os
import subprocess
import time
import warnings
import getpass
import random

parser = argparse.ArgumentParser(description='Dispatch a list of python jobs from a given file to the CML cluster')

# Central:
parser.add_argument('file', type=argparse.FileType())

parser.add_argument('--conda', default='/cmlscratch/jonas0/miniconda3/envs/dl', type=str, help='Path to conda env')
parser.add_argument('--email', default=None, type=str, help='Your email if not @umiacs.umd.edu')
parser.add_argument('--min_preemptions', action='store_true', help='Launch only on nodes where this user has no scavenger jobs.')

parser.add_argument('--qos', default='scav', type=str, help='QOS, choose default, medium, high, scav')
parser.add_argument('--name', default=None, type=str, help='Name that will be displayed in squeue. Default: file name')
parser.add_argument('--gpus', default='1', type=int, help='Requested GPUs PER job')
parser.add_argument('--mem', default='32', type=int, help='Requested memory PER job')
parser.add_argument('--timelimit', default=72, type=int, help='Requested hour limit PER job')
parser.add_argument('--throttling', default=None, type=int, help='Launch only this many jobs concurrently')


args = parser.parse_args()


# Parse and validate input:
if args.name is None:
    dispatch_name = args.file.name
else:
    dispatch_name = args.name

args.conda = args.conda.rstrip('/')

# Usage warnings:
if args.mem > 385:
    raise ValueError('Maximal node memory exceeded.')
if args.gpus > 8:
    raise ValueError('Maximal node GPU number exceeded.')
if args.qos == 'high' and args.gpus > 4:
    warnings.warn('QOS only allows for 4 GPUs, GPU request has been reduced to 4.')
    args.gpus = 4
if args.qos == 'medium' and args.gpus > 2:
    warnings.warn('QOS only allows for 2 GPUs, GPU request has been reduced to 2.')
    args.gpus = 2
if args.qos == 'default' and args.gpus > 1:
    warnings.warn('QOS only allows for 1 GPU, GPU request has been reduced to 1.')
    args.gpus = 1
if args.mem / args.gpus > 48:
    warnings.warn('You are oversubscribing to memory. '
                  'This might leave some GPUs idle as total node memory is consumed.')
if args.qos == 'high' and args.timelimit > 48:
    warnings.warn('QOS only allows for 48 hours. Timelimit request has been reduced to 48.')
    args.gpus = 4

# 1) Strip the file_list of comments and blank lines
content = args.file.readlines()
jobs = [c.strip().split('#', 1)[0] for c in content if 'python' in c and c[0] != '#']

print(f'Detected {len(jobs)} jobs.')

# Write the clean file list
authkey = random.randint(10**5, 10**6 - 1)
with open(f".cml_job_list_{authkey}.temp.sh", "w") as file:
    file.writelines(chr(10).join(job for job in jobs))
    file.write("\n")


# 2) Decide which nodes not to use in scavenger:
username = getpass.getuser()
safeguard = [username]

all_nodes = set(f'cml{i:02}' for i in range(15)) | set(f'cmlgrad0{i}' for i in range(8))
banned_nodes = set()


if args.min_preemptions:
    try:
        raw_status = subprocess.run("squeue", capture_output=True)
        cluster_status = [s.split() for s in str(raw_status.stdout).split('\\n')]
        for sjob in cluster_status[1:-1]:
            if sjob[1] == 'scavenger' and sjob[3] in safeguard and 'cml' in sjob[-1]:
                banned_nodes.add(sjob[-1])
    except FileNotFoundError:
        print('Node exclusion only works when called on cml nodes.')
node_list = sorted(all_nodes - banned_nodes)
banned_nodes = sorted(banned_nodes)

# 3) Prepare environment
if not os.path.exists('log'):
    os.makedirs('log')

# 4) Construct the sbatch launch file
if args.qos == 'scav':
    cml_account = 'scavenger'
elif args.qos in ['high', 'very_high']:
    cml_account = 'tomg'
else:
    cml_account = 'cml'

SBATCH_PROTOTYPE = \
    f"""#!/bin/bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name={''.join(e for e in dispatch_name if e.isalnum())}
#SBATCH --array={f"1-{len(jobs)}" if args.throttling is None else f"1-{len(jobs)}%{args.throttling}"}
#SBATCH --output=log/%x_%A_%a.log
#SBATCH --error=log/%x_%A_%a.log
#SBATCH --time={args.timelimit}:00:00
#SBATCH --account={cml_account}
#SBATCH --qos={args.qos if args.qos != "scav" else "scavenger"}
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --cpus-per-task={min(args.gpus * 4, 32)}
#SBATCH --partition={"dpart" if args.qos != "scav" else "scavenger"}
{f"#SBATCH --exclude={','.join(str(node) for node in banned_nodes)}" if banned_nodes else ''}
#SBATCH --mem={args.mem}gb
#SBATCH --mail-user={args.email if args.email is not None else username + "@umiacs.umd.edu"}
#SBATCH --mail-type=FAIL,ARRAY_TASKS

source {"/".join(args.conda.split("/")[:-2])}/etc/profile.d/conda.sh
conda activate {args.conda}

export MASTER_PORT=$(shuf -i 2000-65000 -n 1)
export MASTER_ADDR=`/bin/hostname -s`

srun $(head -n $((${{SLURM_ARRAY_TASK_ID}})) .cml_job_list_{authkey}.temp.sh | tail -n 1)

"""


# cleanup:
# rm .cml_job_list_{authkey}.temp.sh
# rm .cml_launch_{authkey}.temp.sh
# this breaks SLURM in some way?

# Write launch commands to file
with open(f".cml_launch_{authkey}.temp.sh", "w") as file:
    file.write(SBATCH_PROTOTYPE)


# 5) Print launch information

print('Launch prototype is ...')
print('---------------')
print(SBATCH_PROTOTYPE)
print('---------------')
print(chr(10).join('srun ' + job for job in jobs))
print(f'Preparing {len(jobs)} jobs as user {username}'
      f' for launch on nodes {",".join(str(node) for node in node_list)} in 10 seconds...')
print('Terminate if necessary ...')
for _ in range(10):
    time.sleep(1)

# 6) Launch

# Execute file with sbatch
subprocess.run(["/usr/bin/sbatch", f".cml_launch_{authkey}.temp.sh"])
print('Subprocess launched ...')
time.sleep(3)
subprocess.run("squeue")
