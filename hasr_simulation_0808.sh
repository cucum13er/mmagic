#!/bin/bash
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu-long  # Partition
#SBATCH -G 4 # Number of GPUs
#SBATCH -t 01:00:00 # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

module load cuda/11.7.0
module load miniconda/22.11.1-1
conda activate /work/pi_xiandu_umass_edu/ruima/conda_env/openmmlab/
bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4_backup.py 4 --work-dir work_dirs/restorers/hasr/X4/hasr_0808/

