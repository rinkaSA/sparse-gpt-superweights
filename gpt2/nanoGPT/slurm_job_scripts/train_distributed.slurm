#!/bin/bash
#SBATCH --job-name=lth_gpt2
#SBATCH --nodes=2    
#SBATCH --ntasks-per-node=1    
#SBATCH --gres=gpu:4           
#SBATCH --cpus-per-task=8 
#SBATCH --mem=64G       
#SBATCH --time=24:00:00 
#SBATCH --output=train_out/lth_train_%j.out
#SBATCH --error=train_out/lth_train_%j.err

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=13245

source /software/rome/r24.04/Miniconda3/24.7.1-0/bin/activate universal-ner

cd /data/horse/ws/irve354e-energy_llm_ner/super_weights/gpt2/nanoGPT

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5

WORLD_SIZE=$((SLURM_NNODES * 4))  

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=4 \
    --rdzv-id=$SLURM_JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    train.py \
    --max_iters=10000\
    --out_dir="train_out/lth_iter_${SLURM_JOB_ID}"
