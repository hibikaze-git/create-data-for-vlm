#!/bin/bash

# Command line options go here
#SBATCH --time=7:45:00
#SBATCH --nodelist=slurm0-a3-ghpc-6
#SBATCH --job-name=vlm_synthesis_data
#SBATCH --output=sbatch_logs/3_least_dim_range=512-768.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100GB
#--begin=2024-06-22T15:20:06

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp

cd /storage4/work/yamaguchi/create-data-for-vlm
bash tools/update_0618cc_gpu_num.sh slurm0-a3-ghpc-6 stop 1
#echo "bash tools/update_0618cc_gpu_num.sh slurm0-a3-ghpc-6 stop 1" | at 20:00 21.06.2024

cd /storage4/work/yamaguchi/create-data-for-vlm/datasets/commoncatalog-cc-by-sa

python synthesis_data.py ./commoncatalog-cc-by-sa-download/3/least_dim_range=512-768 ./images/3/least_dim_range=512-768 ./jsonl/3/least_dim_range=512-768
