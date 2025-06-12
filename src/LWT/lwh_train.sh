#!/bin/bash
#SBATCH --job-name=lth_hf_gpt2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=lth_hf_out_%j.out
#SBATCH --error=lth_hf_err_%j.err

FIRST_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(srun -N1 -n1 -w $FIRST_HOST hostname -I | awk '{print $1}')
MASTER_PORT=12345


source /software/rome/r24.04/Miniconda3/24.7.1-0/bin/activate universal-ner


export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5

export NCCL_SOCKET_IFNAME=ib0 

WORLD_SIZE=$((SLURM_NNODES * 4))

srun --nodes=$SLURM_NNODES \
     --ntasks-per-node=1 \
     --output=/dev/null \
     --error=/dev/null \
     bash -lc 'echo "$SLURMD_NODENAME: $(ip -4 addr show dev ib0 \
                                         | awk "/inet/ {print \$2}" \
                                         | cut -d/ -f1)"'
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=4 \
    --rdzv-id=$SLURM_JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    train_lth_sw.py \

