#!/bin/bash
#SBATCH --job-name=sw
#SBATCH --output=logs/superweights_%j.out
#SBATCH --error=logs/superweights_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G


source /software/rome/r24.04/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate universal-ner

python run_app.py
