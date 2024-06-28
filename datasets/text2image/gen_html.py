"""
textを素にhtmlを生成

python gen_html.py stockmark/ner-wikipedia-dataset
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


def list_jsonl_texts(directory):
    # 指定されたディレクトリ内の.jsonlファイルを取得
    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))

    text_list = []

    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            text_list.append(json.loads(f.readline())["text"])

    return text_list


def main(args):
    phi3_manager = Phi3Manager()

    dataset = load_dataset(
        args.dataset_repo,
        cache_dir=f"./cache/{args.dataset_name}",
    )

    processed_ids = list_jsonl_texts(args.jsonl_output_dir)
    #processed_ids = []
    print("\n==================== processed_ids ====================")
    print(processed_ids)
    print("============================================================\n")

    count = 0

    for i, data in tqdm(enumerate(dataset["train"])):
        text = data["text"]

        if text not in processed_ids:
            output_filename = str(i) + ".jsonl"
            output_path = os.path.join(args.jsonl_output_dir, output_filename)

            try:
                synthesis_dict = phi3_manager.translate(text)
            except torch.cuda.OutOfMemoryError as e:
                print(e)
                continue

            synthesis_dict["text"] = text
            synthesis_dict["dataset_repo"] = args.dataset_repo

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(synthesis_dict, ensure_ascii=False) + "\n")

        else:
            print(f"skip: {text}")

        count += 1

        #if count >= 50000:
        #    print("processd max data num")
        #    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 必須の引数
    parser.add_argument("dataset_repo", type=str, help="dataset repo")

    args = parser.parse_args()

    args.dataset_name = args.dataset_repo.split("/")[-1]

    args.jsonl_output_dir = os.path.join("./jsonl", args.dataset_name)

    # 出力ディレクトリの作成
    make_dir(args.jsonl_output_dir)

    # メイン関数の実行
    main(args)

    print("==================================")
    print("======== process complete ========")
    print("==================================")
