#!/bin/bash

# Command line options go here
#SBATCH --time=10:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-15
#SBATCH --job-name=vlm_translate_data
#SBATCH --output=sbatch_logs/intergps.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100GB
# --begin=2024-06-22T06:00:02

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp

cd /storage4/work/yamaguchi/create-data-for-vlm
bash tools/update_0618cc_gpu_num.sh slurm0-a3-ghpc-15 stop 1
#echo "bash tools/update_0618cc_gpu_num.sh slurm0-a3-ghpc-15 stop 1" | at 1:10 22.06.2024

cd /storage4/work/yamaguchi/create-data-for-vlm/datasets/cauldron

python translate.py intergps
