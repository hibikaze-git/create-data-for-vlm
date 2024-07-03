#!/bin/bash

# Command line options go here
#SBATCH --time=7:15:00
#SBATCH --nodelist=slurm0-a3-ghpc-1
#SBATCH --job-name=vlm_t2h
#SBATCH --output=sbatch_logs/wikipedia-22-12-ja-embeddings.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100GB
# --begin=2024-06-22T06:00:02

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp

cd /storage4/work/yamaguchi/create-data-for-vlm
bash tools/update_0618cc_gpu_num.sh slurm0-a3-ghpc-1 stop 1
#echo "bash tools/update_0618cc_gpu_num.sh slurm0-a3-ghpc-1 stop 1" | at 1:10 22.06.2024

cd /storage4/work/yamaguchi/create-data-for-vlm/datasets/text2image

python gen_html.py Cohere/wikipedia-22-12-ja-embeddings
