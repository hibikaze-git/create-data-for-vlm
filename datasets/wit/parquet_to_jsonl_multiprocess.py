"""
parquet→jsonl化

.parquetファイルの名称に重複がないか確認が必要

python parquet_to_jsonl.py input_dir output_dir
"""

import argparse
import glob
import json
import os
import shutil
from multiprocessing import Pool

from datasets import load_dataset

CACHE_PATH = "./dataset_cache"


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def rm_cache(cache_path):
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


def process_file(file_path, output_dir, processed_file_paths_path):
    print(file_path)
    filename = os.path.basename(file_path).replace(".parquet", "")

    pid = os.getpid()
    cache_dir = os.path.join(CACHE_PATH, f"dataset_cache_{pid}")

    dataset = load_dataset(
        "parquet", data_files=file_path, split="train", cache_dir=cache_dir
    )

    output_filename = filename + ".jsonl"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w") as f:
        for i, data in enumerate(dataset):
            language_list = data["wit_features"]["language"]

            if "ja" in language_list:
                image_name = f"{filename}_{i}.jpg"
                #data["image"].save(f"./images/{image_name}")
                with open(f"./images/{image_name}", "wb") as file:
                    file.write(data["image"]["bytes"])

                del data["image"]

                data["image_name"] = image_name
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    with open(processed_file_paths_path, "a") as f:
        f.write(file_path + "\n")

    rm_cache(cache_dir)


def main(args):
    pattern = os.path.join(args.input_dir, "**/*.parquet")
    file_paths = glob.glob(pattern, recursive=True)
    print("num_file_path:", len(file_paths))
    print(file_paths[:5])

    target_file_paths = file_paths

    processed_file_paths_filename = "processed_file_paths.txt"
    processed_file_paths_path = os.path.join(
        args.output_dir, processed_file_paths_filename
    )

    if os.path.exists(processed_file_paths_path):
        processed_file_paths = []

        with open(processed_file_paths_path, "r") as f:
            for line in f:
                # 各行から改行文字を削除
                stripped_line = line.strip()
                # 改行文字が削除された行をリストに追加
                processed_file_paths.append(stripped_line)

        print(processed_file_paths[:5])

        target_file_paths = [
            item for item in file_paths if item not in processed_file_paths
        ]
    else:
        with open(processed_file_paths_path, "w") as f:
            f.write("")

    print(target_file_paths[:5])

    with Pool(processes=os.cpu_count()) as pool:
        results = pool.starmap(
            process_file,
            [
                (path, args.output_dir, processed_file_paths_path)
                for path in target_file_paths
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 必須の引数
    parser.add_argument("input_dir", type=str, help="Path to the input directory")
    parser.add_argument("output_dir", type=str, help="Path to the output directory")

    args = parser.parse_args()

    # 出力ディレクトリの作成
    make_dir(args.output_dir)

    # メイン関数の実行
    main(args)
