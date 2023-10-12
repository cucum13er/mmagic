#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))

set -x

CONFIG=$1
WORK_DIR=$2
GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
PY_ARGS=${@:5}
SRUN_ARGS=${SRUN_ARGS:-""}

#PYTHONPATH="/work/pi_xiandu_umass_edu/ruima/git/mmediting_Rui_git/" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun --export=ALL \
     --gres=gpu:${GPUS_PER_NODE} \
     --ntasks=${GPUS} \
     --ntasks-per-node=${GPUS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_TASK} \
     --kill-on-bad-exit=1 \
     ${SRUN_ARGS} \
     python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}
