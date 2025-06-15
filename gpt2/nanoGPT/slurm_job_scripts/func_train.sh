#!/bin/bash
#SBATCH --job-name=lth_hf_gpt2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=lth_hf_out_%j.out
#SBATCH --error=lth_hf_err_%j.err

FIRST_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(srun -N1 -n1 -w $FIRST_HOST hostname -I | awk '{print $1}')
MASTER_PORT=12345


source /software/rome/r24.04/Miniconda3/24.7.1-0/bin/activate universal-ner

#rm -rf ~/.triton/cache -. use it if any issues with triton, glib versioon mismatch!
#rm -rf ~/.cache/torch/inductor
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5

export NCCL_SOCKET_IFNAME=ib0 

WORLD_SIZE=$((SLURM_NNODES * 4))

cd /data/horse/ws/irve354e-energy_llm_ner/super_weights/gpt2/nanoGPT

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=4 \
    --rdzv-id=$SLURM_JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    slurm_job_scripts/func_train.py \

