#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu-long  # Partition
#SBATCH -G 4 # Number of GPUs
#SBATCH -N 1 # ONE NODE
#SBATCH -t 06-23 # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

module load cuda/11.7.0
module load miniconda/22.11.1-1
conda activate /work/pi_xiandu_umass_edu/ruima/conda_env/openmmlab/
GPUS=4 GPUS_PER_NODE=4 CPUS_PER_TASK=1 ./tools/slurm_train_Rui.sh configs/restorers/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost_Rui_basicX2.py work_dirs/restorers/swinir/X2/basic/

