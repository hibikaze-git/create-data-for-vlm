"""
python upload_image_folder.py --num_workers 8
"""

import argparse
import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from datasets import load_dataset

ORIGIN_IMAGE_DIR = "./images"
IMAGE_DIR = "./images_dpo"

INSTRUCTION_POOLS = [
    "画像に写っているものは何ですか？",
    "写真に映っているものは何ですか。",
    "画像に関する情報を提示してください",
    "写真に写っている物の名前と関連する情報",
    "写真に写っているものは何か。",
    "写真の中にある物が何かを関連知識と一緒に示してください。",
    "写真に映っている物の関連情報",
    "写真の中の物が何か、名前と関連する背景知識を提示してください。",
    "写真に写っている物が何かを関連知識と合わせて示してください。",
    "画像に写っているものの背景情報",
    "画像に含まれている物が何であるか",
]


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_files(directory, ext):
    for filename in os.listdir(directory):
        if filename.endswith(f".{ext}"):
            filepath = os.path.join(directory, filename)
            os.remove(filepath)


def create_metadata(data):
    i, data_str = data

    data_dict = json.loads(data_str)

    wit_features = data_dict["wit_features"]
    language_list = wit_features["language"]

    results = []

    for i, lang in enumerate(language_list):
        content = {}

        if lang == "ja":
            hierarchical_section_title = wit_features["hierarchical_section_title"][i]
            page_title = wit_features["page_title"][i]

            if hierarchical_section_title == page_title:
                image_filename = data_dict["image_name"]

                answer = wit_features["context_page_description"][i]

                if answer:
                    shutil.copy(os.path.join(ORIGIN_IMAGE_DIR, image_filename), IMAGE_DIR)

                    answer = answer.strip().replace("\n", "")

                    content["file_name"] = image_filename
                    content["question"] = random.choice(INSTRUCTION_POOLS)
                    content["chosen"] = answer
                    content["rejected"] = page_title

                    results.append(content)

    return results


def main(num_workers):
    # データセットのリストを作成
    with open("wit_base.jsonl", "r", encoding="utf-8") as f:
        dataset = f.readlines()

    dataset_list = [(i, content) for i, content in enumerate(dataset)]

    print("=========== data len ==========")
    print(len(dataset_list))
    print("===============================")

    metadata_list = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(create_metadata, data) for data in dataset_list]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                metadata_list.extend(result)

    return metadata_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process HTML to images with multiprocessing."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes to use.",
    )
    args = parser.parse_args()

    make_dir(IMAGE_DIR)

    delete_files(IMAGE_DIR, "jsonl")

    metadata_list = main(args.num_workers)

    with open(f"{IMAGE_DIR}/metadata.jsonl", "w", encoding="utf-8") as f:
        for html_text in metadata_list:
            f.write(json.dumps(html_text, ensure_ascii=False) + "\n")

    upload_dataset = load_dataset(
        "imagefolder", data_dir=IMAGE_DIR, cache_dir="./cache"
    )
    upload_dataset.push_to_hub(
        "hibikaze/wit-dpo-test", private=False
    )
