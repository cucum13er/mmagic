#!/bin/bash
#SBATCH --partition=gypsum-m40
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=16G
#SBATCH --constraint="vram16"
#SBATCH --time=5:00:00
#SBATCH -o slurm-%j.out  # %j = job ID
module load cuda/11.7.0
module load miniconda/22.11.1-1
conda activate /work/pi_xiandu_umass_edu/ruima/conda_env/openmmlab/
GPUS=4 GPUS_PER_NODE=4 CPUS_PER_TASK=1 ./tools/slurm_train_Rui.sh configs/restorers/restormer/restormer_Rui_original_DRealSR_X2.py work_dirs/restorers/restormer/X2/pretrain_DRealSR/ 
