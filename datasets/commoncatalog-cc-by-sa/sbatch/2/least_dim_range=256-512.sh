#!/bin/bash

# Command line options go here
#SBATCH --time=0:45:00
#SBATCH --nodelist=slurm0-a3-ghpc-2
#SBATCH --job-name=vlm_synthesis_data
#SBATCH --output=sbatch_logs/2_least_dim_range=256-512.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100GB
#SBATCH --begin=2024-06-22T02:15:00

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp

cd /storage4/work/yamaguchi/create-data-for-vlm
bash tools/update_0618cc_gpu_num.sh slurm0-a3-ghpc-2 stop 1
#echo "bash tools/update_0618cc_gpu_num.sh slurm0-a3-ghpc-2 stop 1" | at 20:00 21.06.2024

cd /storage4/work/yamaguchi/create-data-for-vlm/datasets/commoncatalog-cc-by-sa

python synthesis_data.py ./commoncatalog-cc-by-sa-download/2/least_dim_range=256-512 ./images/2/least_dim_range=256-512 ./jsonl/2/least_dim_range=256-512