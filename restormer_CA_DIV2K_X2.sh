#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=16G
#SBATCH --time=6-23
#SBATCH -o slurm-%j.out  # %j = job ID
module load cuda/11.7.0
module load miniconda/22.11.1-1
conda activate /work/pi_xiandu_umass_edu/ruima/conda_env/openmmlab/
GPUS=4 GPUS_PER_NODE=4 CPUS_PER_TASK=1 ./tools/slurm_train_Rui.sh configs/restorers/restormer/restormer_Rui_CA_DIV2K_X2.py work_dirs/restorers/restormer/X2/CA_DIV2K/ 

