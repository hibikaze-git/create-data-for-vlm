#!/bin/bash

# Command line options go here
#SBATCH --time=96:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-0
#SBATCH --job-name=dl_data
#SBATCH --output=sbatch_logs/download_wit_base.out
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#--begin=2024-07-03T16:00:00

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp

cd /storage5/multimodal/work/yamaguchi/create-data-for-vlm
bash tools/update_0705cc_gpu_num.sh slurm0-a3-ghpc-0 stop 1

cd /storage5/multimodal/work/yamaguchi/create-data-for-vlm/datasets/wit

python download_wit_base.py
