#!/bin/bash
#SBATCH --job-name=nanogpt_prep
#SBATCH --output=slurm_out/checkmodel_%j.out
#SBATCH --error=slurm_out/checkmodel_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G


source /software/rome/r24.04/Miniconda3/24.7.1-0/bin/activate universal-ner


cd /data/horse/ws/irve354e-energy_llm_ner/super_weights/super_weights/src

python main.py --model_name gpt2

conda deactivate
