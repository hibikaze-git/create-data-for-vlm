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
        with open(jsonl_file, "r", encoding="utf-8") as f:
            text_list.append(json.loads(f.readline())["text"])

    return text_list


def main(args):
    phi3_manager = Phi3Manager()

    dataset = load_dataset(
        args.dataset_repo,
        cache_dir=f"./cache/{args.dataset_name}",
    )

    processed_ids = list_jsonl_texts(args.jsonl_output_dir)
    # processed_ids = []
    print("\n==================== processed_ids ====================")
    print(processed_ids)
    print("============================================================\n")

    count = 0
    batch_num = 4
    text_batch = []

    for i, data in tqdm(enumerate(dataset["train"])):
        text = data["text"]

        if text not in processed_ids:
            text_batch.append({"text": text, "id": i})

            if len(text_batch) == batch_num:
                try:
                    synthesis_dict_list = phi3_manager.translate(text_batch)
                except torch.cuda.OutOfMemoryError as e:
                    print(e)
                    continue

                for synthesis_dict in synthesis_dict_list:
                    synthesis_dict["dataset_repo"] = args.dataset_repo

                    output_filename = str(synthesis_dict["id"]) + ".jsonl"
                    output_path = os.path.join(args.jsonl_output_dir, output_filename)

                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(json.dumps(synthesis_dict, ensure_ascii=False) + "\n")

                text_batch = []
        else:
            print(f"skip: {text}")

        count += 1

        # if count >= 50000:
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
