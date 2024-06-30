"""
データ翻訳

python translate_plotqa_shuffle.py plotqa_shuffle 1000-2000
"""

import argparse
import glob
import json
import os

import torch

from phi3 import Phi3Manager
from tqdm import tqdm

from datasets import load_dataset


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def list_jsonl_files(directory):
    # 指定されたディレクトリ内の.jsonlファイルを取得
    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    # ファイル名だけを抽出
    jsonl_files = [
        int(os.path.basename(file).replace(".jsonl", "")) for file in jsonl_files
    ]
    return jsonl_files


def main(args):
    phi3_manager = Phi3Manager(args.subset_name)

    dataset = load_dataset(
        "team-hatakeyama-phase2/the_cauldron_subset_with_id",
        args.subset_name,
        cache_dir=f"./cache/{args.subset_name}",
    )

    processed_ids = list_jsonl_files(args.jsonl_output_dir)
    # processed_ids = []
    print("\n==================== processed_ids ====================")
    print(processed_ids)
    print("============================================================\n")

    [start_id, end_id] = [int(id) for id in args.id_range.split("-")]

    for i, data in tqdm(enumerate(dataset["train"])):
        if (
            data["id"] not in processed_ids
            and data["id"] >= start_id
            and data["id"] < end_id
        ):
            print(data["id"])

            output_filename = str(data["id"]) + ".jsonl"
            output_path = os.path.join(args.jsonl_output_dir, output_filename)

            try:
                synthesis_dict = phi3_manager.translate(data["texts"])
            except torch.cuda.OutOfMemoryError as e:
                print(e)
                continue

            synthesis_dict["id"] = data["id"]
            synthesis_dict["texts"] = data["texts"]

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(synthesis_dict, ensure_ascii=False) + "\n")

        else:
            print(f"skip: {data['id']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 必須の引数
    parser.add_argument("subset_name", type=str, help="subset name")
    parser.add_argument("id_range", type=str, help="target id range")

    args = parser.parse_args()
    args.jsonl_output_dir = os.path.join("./jsonl", args.subset_name)

    # 出力ディレクトリの作成
    make_dir(args.jsonl_output_dir)

    # メイン関数の実行
    main(args)

    print("==================================")
    print("======== process complete ========")
    print("==================================")
