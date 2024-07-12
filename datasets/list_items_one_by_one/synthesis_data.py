"""
データ合成

python synthesis_data.py ./train2017 ./images ./jsonl
"""

import argparse
import glob
import json
import os

from calm3 import Calm3Manager
from florence import FlorenceManager
from phi3 import Phi3Manager
from tqdm import tqdm


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def list_processed_files(directory):
    # 指定されたディレクトリ内のファイルを取得
    files = glob.glob(os.path.join(directory, "*.jsonl"))
    # ファイル名だけを抽出
    files = [os.path.splitext(os.path.basename(file))[0] for file in files]
    return files


def main(args):
    florence_manager = FlorenceManager(args.image_output_dir)
    # phi3_manager = Phi3Manager()
    # calm3_manager = Calm3Manager()

    skip_id_list = [
        "000000194000",
        "000000537918"
    ]

    with open(
        "./annotations/captions_train2017.json",
        "r",
        encoding="utf-8",
    ) as f:
        captions = json.load(f)

    license_dict = {}

    for image_info in captions["images"]:
        license_dict[image_info["file_name"].split(".")[0]] = image_info["license"]

    try:
        pattern = os.path.join(args.input_dir, "**/*.*")
        file_paths = glob.glob(pattern, recursive=True)
        print("num_file_path:", len(file_paths))
        print("num_captions:", len(license_dict.keys()))
        print(file_paths[:5])

        target_file_paths = file_paths

        processed_photoids = list_processed_files(args.jsonl_output_dir)
        print("\n==================== processed_photoids ====================")
        print(processed_photoids)
        print("============================================================\n")

        for i, file_path in tqdm(enumerate(target_file_paths)):
            print(file_path)
            photoid = os.path.splitext(os.path.basename(file_path))[0]

            if photoid not in processed_photoids and license_dict[photoid] in [4, 5] and photoid not in skip_id_list:
                output_filename = str(photoid) + ".jsonl"
                output_path = os.path.join(args.jsonl_output_dir, output_filename)

                synthesis_dict = florence_manager.synthesis(file_path)

                # synthesis_dict = phi3_manager.translate(synthesis_dict)
                # synthesis_dict = calm3_manager.translate(synthesis_dict)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(synthesis_dict, ensure_ascii=False) + "\n")

            else:
                print(f"skip: {photoid}")

            # if i == 10:
            #    break

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

    args = parser.parse_args()

    # 出力ディレクトリの作成
    make_dir(args.image_output_dir)
    make_dir(args.jsonl_output_dir)

    # メイン関数の実行
    main(args)

    print("==================================")
    print("======== process complete ========")
    print("==================================")
