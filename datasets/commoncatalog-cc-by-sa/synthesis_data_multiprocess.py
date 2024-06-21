"""
データ合成

python synthesis_data_multiprocess.py ./commoncatalog-cc-by-sa-download/0 ./images/0 ./jsonl/0 --num_processes 3
python synthesis_data_multiprocess.py ./commoncatalog-cc-by-sa-download/0/least_dim_range=768-1024 ./images/0/least_dim_range=768-1024 ./jsonl/0/least_dim_range=768-1024 --num_processes 2
python synthesis_data_multiprocess.py ./commoncatalog-cc-by-sa-download/1/least_dim_range=1024-2048 ./images/1/least_dim_range=1024-2048 ./jsonl/1/least_dim_range=1024-2048 --num_processes 2
"""

import argparse
import glob
import json
import os
import shutil
from datetime import datetime
from multiprocessing import Manager, Pool, cpu_count

from phi3 import Phi3Manager
from phi3_vision import Phi3VisionManager
from tqdm import tqdm

from datasets import load_dataset


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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


def copy_json_file_with_timestamp(src, dst_dir):
    try:
        # 現在時刻を取得してフォーマット
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # コピー元のファイル名を取得
        base_name = os.path.basename(src)

        # 拡張子を分離
        name, ext = os.path.splitext(base_name)

        # 新しいファイル名を作成
        new_file_name = f"{name}_{timestamp}{ext}"

        # コピー先のパスを作成
        dst = os.path.join("bk", dst_dir, new_file_name)

        # ファイルをコピー
        shutil.copy(src, dst)
        print(f"File copied successfully from {src} to {dst}")
    except IOError as e:
        print(f"Unable to copy file. {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def process_file_wrapper(args_lock_paths_fp):
    args, lock, processed_file_paths, processed_file_paths_path, fp = args_lock_paths_fp
    return process_file(fp, args, lock, processed_file_paths, processed_file_paths_path)


def process_file(
    file_path, args, lock, processed_file_paths, processed_file_paths_path
):
    phi3_vision_manager = Phi3VisionManager()
    phi3_manager = Phi3Manager()

    try:
        print(file_path)
        filename = os.path.basename(file_path).replace(".parquet", "")

        pid = os.getpid()
        cache_dir = f"dataset_cache_{pid}"

        dataset = load_dataset(
            "parquet", data_files=file_path, split="train", cache_dir=cache_dir
        )

        output_filename = filename + ".jsonl"
        output_path = os.path.join(args.jsonl_output_dir, output_filename)

        jsonl_dataset = []

        for i, data in tqdm(enumerate(dataset)):
            image_output_path = os.path.join(
                args.image_output_dir, f"{data['photoid']}.{data['ext']}"
            )

            with open(image_output_path, "wb") as file:
                file.write(data["jpg"])

            synthesis_dict = phi3_vision_manager.synthesis(image_output_path)

            synthesis_dict.update(phi3_manager.translate(synthesis_dict))

            synthesis_dict["photoid"] = data["photoid"]
            synthesis_dict["ext"] = data["ext"]

            jsonl_dataset.append(synthesis_dict)

        with open(output_path, "w", encoding="utf-8") as f:
            for i, data in enumerate(jsonl_dataset):
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        with lock:
            processed_file_paths.append(file_path)
            with open(processed_file_paths_path, "w") as f:
                f.write(json.dumps(processed_file_paths, ensure_ascii=False))

        rm_cache(cache_dir)
        return file_path
    except Exception as e:
        print(e)
        rm_cache(cache_dir)
        return None


def main(args):
    try:
        pattern = os.path.join(args.input_dir, "**/*.parquet")
        file_paths = glob.glob(pattern, recursive=True)
        print("num_file_path:", len(file_paths))
        print(file_paths[:5])

        split_input_dir = args.input_dir.split("/")

        if len(split_input_dir) <= 3:
            processed_file_paths_filename = (
                f"processed_file_paths_{split_input_dir[-1]}.json"
            )
        else:
            processed_file_paths_filename = (
                f"processed_file_paths_{split_input_dir[-2] + '_' + split_input_dir[-1]}.json"
            )

        processed_file_paths_path = os.path.join(processed_file_paths_filename)
        processed_file_paths = []

        if os.path.exists(processed_file_paths_path):
            copy_json_file_with_timestamp(processed_file_paths_path, "./")

            with open(processed_file_paths_path, "r") as f:
                processed_file_paths = json.load(f)

            print(processed_file_paths[:5])

        target_file_paths = [
            item for item in file_paths if item not in processed_file_paths
        ]

        print(target_file_paths[:5])

        manager = Manager()
        lock = manager.Lock()
        processed_file_paths = manager.list(processed_file_paths)

        with Pool(processes=args.num_processes) as pool:
            for _ in tqdm(
                pool.imap_unordered(
                    process_file_wrapper,
                    [
                        (
                            args,
                            lock,
                            processed_file_paths,
                            processed_file_paths_path,
                            fp,
                        )
                        for fp in target_file_paths
                    ],
                )
            ):
                pass

    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 必須の引数
    parser.add_argument("input_dir", type=str, help="Path to the input directory")
    parser.add_argument(
        "image_output_dir", type=str, help="Path to the image output directory"
    )
    parser.add_argument(
        "jsonl_output_dir", type=str, help="Path to the jsonl output directory"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=cpu_count(),
        help="Number of processes to use (default: number of CPU cores)",
    )

    args = parser.parse_args()

    # 出力ディレクトリの作成
    make_dir(args.image_output_dir)
    make_dir(args.jsonl_output_dir)

    # メイン関数の実行
    main(args)

    print("==================================")
    print("======== process complete ========")
    print("==================================")
