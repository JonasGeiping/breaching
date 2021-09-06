"""Launch a jupyter notebook on SLURM.

Before running this script, run
jupyter

and


on CML to initialize your jupyter notebook and set your password.

"""



import argparse
import secrets

import subprocess
import time
import warnings
import getpass

parser = argparse.ArgumentParser(description='Launch a jupyter notebook on the CML cluster')
# Central:
parser.add_argument('--conda', default='/cmlscratch/jonas0/miniconda3/envs/dl', type=str, help='Path to conda env')
parser.add_argument('--qos', default='default', type=str, help='QOS, choose default, medium, high, scav')
parser.add_argument('--gpus', default='1', type=int, help='Requested GPUs PER job')
parser.add_argument('--mem', default='32', type=int, help='Requested memory PER job')
parser.add_argument('--timelimit', default=8, type=int, help='Requested hour limit PER job')
args = parser.parse_args()

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

username = getpass.getuser()
token = secrets.token_urlsafe(10)

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
#SBATCH --job-name=jupyter
#SBATCH --time={args.timelimit}:00:00
#SBATCH --account={cml_account}
#SBATCH --qos={args.qos if args.qos != "scav" else "scavenger"}
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --cpus-per-task={min(args.gpus * 4, 32)}
#SBATCH --partition={"dpart" if args.qos != "scav" else "scavenger"}
#SBATCH --mem={args.mem}gb

source {"/".join(args.conda.split("/")[:-2])}/etc/profile.d/conda.sh
conda activate {args.conda}

export JUPYTER_PORT=$(shuf -i 2000-65000 -n 1)
export HOSTNAME=`/bin/hostname -s`

jupyter notebook --no-browser --port=$JUPYTER_PORT --ip $HOSTNAME --NotebookApp.token={token}

echo -e "
Run this command for the ssh connection:
ssh -N -f -L localhost:$(JUPYTER_PORT):$(HOSTNAME):$(JUPYTER_PORT) {username}@$cmlsub00.umiacs.umd.edu

and open the following web adress in your local browser:
http://localhost:$(JUPYTER_PORT)/?token={token}
"
"""

# Write launch commands to file
with open(f".cml_launch_{authkey}.temp.sh", "w") as file:
    file.write(SBATCH_PROTOTYPE)


# 5) Print launch information

print('Launch prototype is ...')
print('---------------')
print(SBATCH_PROTOTYPE)
print('---------------')
print(f'Preparing jupyter job as user {username}')

# Execute file with sbatch
subprocess.run(["/usr/bin/sbatch", f".cml_launch_{authkey}.temp.sh"])
print('Subprocess launched ...')
time.sleep(3)
subprocess.run("squeue")
