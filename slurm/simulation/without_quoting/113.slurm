#!/bin/bash
#SBATCH --job-name=jae_sim # short name for your job
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=356000 # in megabytes, default is 4GB per task
#SBATCH -o /home/jvavra/TBIP/out/jae_sim/without_quoting/%x_%j_%N.out      # save stdout to file. The filename is defined through filename pattern
#SBATCH -e /home/jvavra/TBIP/err/jae_sim/without_quoting/%x_%j_%N.err      # save stderr to file. The filename is defined through filename pattern
#SBATCH --time=11:59:59 # total run time limit (HH:MM:SS)

. /opt/apps/2023-04-11_lmod.bash
ml purge
ml miniconda3-4.10.3-gcc-12.2.0-ibprkvn

cd /home/jvavra/TBIP/
conda activate tf_1_15

python3 /home/jvavra/TBIP/python/jae_sim/03_variational_initialization_beta_eta.py  --session=113  --scenario=  --reest_seed=0  --data_name=jae_sim  --version=without_quoting  --eps=1  --learningrate=0.0001  --numtopics=25  --batchsize=512  --maxsteps=250000  --printsteps=25000
