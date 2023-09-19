#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 2  # Number of GPUs
#SBATCH -t 01:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

module load cuda/11.7.0
module load miniconda/22.11.1-1
conda activate /work/pi_xiandu_umass_edu/ruima/conda_env/openmmlab/
bash ./tools/dist_train.sh configs/restorers/hasr/hasr_div2kflickr2k_contrastive_MoCo_both_X4.py 1 --work-dir work_dirs/restorers/hasr/X4/hasr_0808/



