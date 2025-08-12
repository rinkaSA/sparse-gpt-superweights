#!/bin/bash
#SBATCH --job-name=NAN_1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=28:00:00
#SBATCH --output=slurm_out/NAN_1_out_%j.log
#SBATCH --error=slurm_out/NAN_1_err_%j.log


SW_LAYER=2
SW_C_OUT=447
SW_C_IN=666
INIT_TYPE='xavier'
PRUNE_PERCENT=10
OUTPUT_DIR='NAN_1'


FIRST_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(srun -N1 -n1 -w $FIRST_HOST hostname -I | awk '{print $1}')
MASTER_PORT=12345


source /software/rome/r24.04/Miniconda3/24.7.1-0/bin/activate universal-ner

#rm -rf ~/.triton/cache -. use it if any issues with triton, glib versioon mismatch!
#rm -rf ~/.cache/torch/inductor

unset NCCL_SOCKET_IFNAME
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
#export NCCL_IB_HCA=mlx5

#export NCCL_SOCKET_IFNAME=ib0 for capella this does not work

WORLD_SIZE=$((SLURM_NNODES * 4))

cd /data/horse/ws/irve354e-energy_llm_ner/super_weights/
export PYTHONPATH=$(pwd)

export $(grep -v '^#' .env | xargs)

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=4 \
    --rdzv-id=$SLURM_JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    LWH/src/func_train.py \
    --sw_layer $SW_LAYER \
    --sw_c_out $SW_C_OUT \
    --sw_c_in $SW_C_IN \
    --init_type $INIT_TYPE \
    --prune_percent $PRUNE_PERCENT \
    --out_dir $OUTPUT_DIR \
    --max_iters 10 \
