#!/bin/bash
#SBATCH --job-name=nanogpt_prep
#SBATCH --output=logs/prepare_data_%j.out
#SBATCH --error=logs/prepare_data_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G


source /software/rome/r24.04/Miniconda3/24.7.1-0/bin/activate universal-ner

mkdir -p logs

cd /data/horse/ws/irve354e-energy_llm_ner/super_weights/gpt2/nanoGPT

python data/openwebtext/prepare.py

conda deactivate
