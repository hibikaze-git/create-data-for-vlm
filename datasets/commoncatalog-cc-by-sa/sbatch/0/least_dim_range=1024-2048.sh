#!/bin/bash

# Command line options go here
#SBATCH --time=8:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-0
#SBATCH --job-name=vlm_synthesis_data
#SBATCH --output=sbatch_logs/0_least_dim_range=1024-2048.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100GB
#SBATCH --begin=2024-06-21T17:00:00

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp

cd /storage4/work/yamaguchi/create-data-for-vlm
#bash tools/update_0618cc_gpu_num.sh slurm0-a3-ghpc-0 stop 1
echo "bash tools/update_0618cc_gpu_num.sh slurm0-a3-ghpc-0 stop 1" | at 17:00 21.06.2024

cd /storage4/work/yamaguchi/create-data-for-vlm/datasets/commoncatalog-cc-by-sa

python synthesis_data.py ./commoncatalog-cc-by-sa-download/0/least_dim_range=1024-2048 ./images/0/least_dim_range=1024-2048 ./jsonl/0/least_dim_range=1024-2048
