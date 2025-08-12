#!/bin/bash
#SBATCH --job-name=lth_gpt2
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=1    
#SBATCH --gres=gpu:1        
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G  
#SBATCH --time=5:00:00 
#SBATCH --output=train_out/single_gpt_train_%j.out
#SBATCH --error=train_out/single_gpt_train_%j.err

rm -rf ~/.triton/cache
rm -rf ~/.cache/torch/inductor

export TRITON_CACHE_DIR=$HOME/triton_cache
mkdir -p $TRITON_CACHE_DIR

source /software/rome/r24.04/Miniconda3/24.7.1-0/bin/activate universal-ner

echo "Running on $(hostname)"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

export PYTHONUNBUFFERED=1

mkdir -p train_out

cd /data/horse/ws/irve354e-energy_llm_ner/super_weights/gpt2/nanoGPT
python train.py \
  --compile=True \
  --max_iters=100
  --out_dir="train_out/sg_${SLURM_JOB_ID}"


