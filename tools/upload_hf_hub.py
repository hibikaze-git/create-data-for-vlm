"""
jsonlをHFにアップロード

huggingface-cli login
python tools/upload_hf_hub.py ./datasets/commoncatalog-cc-by-sa/jsonl/0/part-00000-tid-7761264679195244086-4c6ec9dc-fb3d-4bd3-98ec-f987466013dd-3488916-1-c000.jsonl hibikaze/upload_test
python tools/upload_hf_hub.py ./datasets/commoncatalog-cc-by-sa/jsonl/0/part-00000-tid-7761264679195244086-4c6ec9dc-fb3d-4bd3-98ec-f987466013dd-3488916-1-c000.jsonl hibikaze/upload_test --subset_name subset_name
"""

import argparse
import os
import shutil

from datasets import load_dataset


def rm_cache():
    cache_path = "./dataset_cache"
    permissions = 0o777

    # 権限を再帰的に変更
    for root, dirs, files in os.walk(cache_path):
        for name in dirs:
            dir_path = os.path.join(root, name)
            os.chmod(dir_path, permissions)
        for name in files:
            file_path = os.path.join(root, name)
            os.chmod(file_path, permissions)

    shutil.rmtree(cache_path, ignore_errors=True)


parser = argparse.ArgumentParser(
    description="upload dataset to huggingface hub"
)

parser.add_argument("input_file", type=str, help="Input JSONL file containing text data.")
parser.add_argument("repo_name", type=str, help="hugging face repository name")
parser.add_argument("--subset_name", type=str, help="hugging face repository subset name", default=None)

args = parser.parse_args()

dataset = load_dataset("json", data_files=args.input_file, split="train", cache_dir="./dataset_cache")

if args.subset_name:
    dataset.push_to_hub(args.repo_name, args.subset_name, private=True)
else:
    dataset.push_to_hub(args.repo_name, private=True)

rm_cache()
