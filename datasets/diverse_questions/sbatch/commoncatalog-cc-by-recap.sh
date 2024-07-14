#!/bin/bash

# Command line options go here
#SBATCH --time=23:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-0
#SBATCH --job-name=vlm_synthesis_data
#SBATCH --output=sbatch_logs/commoncatalog-cc-by-recap.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100GB
#--begin=2024-07-03T16:00:00

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm-118

cd /storage5/multimodal/work/yamaguchi/create-data-for-vlm
bash tools/update_0705cc_gpu_num.sh slurm0-a3-ghpc-0 stop 1
#echo "bash tools/update_0618cc_gpu_num.sh slurm0-a3-ghpc-0 stop 1" | at 1:10 22.06.2024

cd /storage5/multimodal/work/yamaguchi/create-data-for-vlm/datasets/diverse_questions

python calm3_batch.py alfredplpl/commoncatalog-cc-by-recap --batch_size 1000 --tensor_parallel_size 1
