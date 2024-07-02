#!/bin/bash

# Command line options go here
#SBATCH --time=7:40:00
#SBATCH --nodelist=slurm0-a3-ghpc-2
#SBATCH --job-name=vlm_translate_data
#SBATCH --output=sbatch_logs/plotq_0-6250.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100GB
# --begin=2024-06-22T06:00:02

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp

cd /storage4/work/yamaguchi/create-data-for-vlm
bash tools/update_0618cc_gpu_num.sh slurm0-a3-ghpc-2 stop 1
#echo "bash tools/update_0618cc_gpu_num.sh slurm0-a3-ghpc-2 stop 1" | at 1:10 22.06.2024

cd /storage4/work/yamaguchi/create-data-for-vlm/datasets/cauldron

python translate_plotqa_shuffle.py plotqa_shuffle 0-6250
