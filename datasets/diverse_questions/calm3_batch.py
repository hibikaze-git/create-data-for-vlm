"""
python calm3_batch.py alfredplpl/commoncatalog-cc-by-recap --batch_size 100 --tensor_parallel_size 4

for test
python calm3_batch.py alfredplpl/commoncatalog-cc-by-recap --num_samples 10 --batch_size 2 --tensor_parallel_size 1
"""

import argparse
import glob
import json
import os
import random

from questions import QUESTIONS
from tqdm import tqdm

import datasets
from vllm import LLM, SamplingParams

TRANSLATE_PROMPT = """\
<|im_start|>system
あなたは親切なAIアシスタントです。<|im_end|>
<|im_start|>user
以下を日本語に翻訳してください。
{text}<|im_end|>
<|im_start|>assistant"""

QA_PROMPT = """\
<|im_start|>system
あなたは親切なAIアシスタントです。<|im_end|>
<|im_start|>user
以下は画像の説明文です。
{scene}

画像の説明に基づき、以下の指示に従ってください。回答は、視覚的な AI アシスタントが画像を見て質問に答えているような口調でなければなりません。
{question}<|im_end|>
<|im_start|>assistant"""


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def list_processed_files(directory):
    # 指定されたディレクトリ内のファイルを取得
    files = glob.glob(os.path.join(directory, "*.jsonl"))
    # id抽出
    filename_list = [os.path.splitext(os.path.basename(file))[0] for file in files]
    return filename_list


def load_dataset(name):
    return datasets.load_dataset(name, cache_dir="./cache")["train"]


class Translater:
    def __init__(self, max_num_seqs, tensor_parallel_size, save_dir) -> None:
        # vLLMでモデルを初期化 tensor_parallel_sizeは使用するGPU数
        self.model = LLM(
            model="cyberagent/calm3-22b-chat",
            # model="Qwen/Qwen2-1.5B-Instruct",  # テスト用
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,  # バッチサイズに合わせて調整
            max_num_batched_tokens=16384,  # トークン数を増やす
            # max_model_len=1024,  # テスト用
            download_dir="../../cache",
        )

        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=1.0,
            max_tokens=2048,
            repetition_penalty=1.05,
        )

        self.save_dir = save_dir

    def translate_batch(self, prompts):
        # print(prompts[0])
        outputs = self.model.generate(prompts, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

    def translate_dataset(self, dataset, batch_size, num_samples):
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        processed_fienames = list_processed_files(self.save_dir)

        if processed_fienames:
            saved_batch_size = int(processed_fienames[0].split("_")[-1])

            assert batch_size == saved_batch_size, "batch_sizeが一致しません。"

        print("\n==================== processed_fienames ====================")
        print(processed_fienames)
        print("============================================================\n")

        for i in tqdm(range(0, len(dataset), batch_size), desc="バッチ処理中"):
            translated_data = []

            filename = f"{i}_{batch_size}"

            if filename in processed_fienames:
                print(f"skip: {filename}")
                continue

            batch = dataset[i : i + batch_size]

            converted_data = []

            target_attr = "phi3_caption"

            for i in range(len(batch[target_attr])):
                item = {key: value[i] for key, value in batch.items()}
                converted_data.append(item)

            # print(converted_data)

            en_texts = batch[target_attr]

            translated_texts = self.translate_batch(
                [TRANSLATE_PROMPT.format(text=text) for text in en_texts]
            )  # 英語から日本語に翻訳

            category_list = random.choices(list(QUESTIONS.keys()), k=batch_size)

            question_list = []

            for category in category_list:
                question_list.append(random.choice(QUESTIONS[category]))

            answer_list = self.translate_batch(
                [
                    QA_PROMPT.format(question=question, scene=translated_text)
                    for question, translated_text in zip(
                        question_list, translated_texts
                    )
                ]
            )

            for data, category, question, answer, translated_text in zip(
                converted_data, category_list, question_list, answer_list, translated_texts
            ):
                data["phi3_caption_ja"] = translated_text
                data["category"] = category
                data["question"] = question
                data["answer"] = answer
                translated_data.append(data)

            with open(
                os.path.join(self.save_dir, f"{filename}.jsonl"), "w", encoding="utf-8"
            ) as f:
                for data in translated_data:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate a dataset using calm3 model"
    )
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to load")
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Number of samples to process"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for translation"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="set num gpu"
    )

    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name)

    save_dir = os.path.join("./jsonl", args.dataset_name.split("/")[-1])
    make_dir(save_dir)

    translater = Translater(args.batch_size, args.tensor_parallel_size, save_dir)

    translater.translate_dataset(
        dataset,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )

    print("==================================")
    print("======== process complete ========")
    print("==================================")
