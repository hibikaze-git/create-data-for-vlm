"""
python calm3_translate_batch.py atsushi3110/en-ja-parallel-corpus-augmented --num_samples 10 --output_file en_ja_parallel_calm3.json --batch_size 100 --tensor_parallel_size 1
"""

import argparse
import json

from tqdm import tqdm

import datasets
from vllm import LLM, SamplingParams

PROMPT = """\
<|im_start|>system
あなたは親切なAIアシスタントです。<|im_end|>
<|im_start|>user
以下を日本語に翻訳してください。
{text}<|im_end|>
<|im_start|>assistant"""


def load_dataset(name):
    return datasets.load_dataset(name)["train"]


class Translater:
    def __init__(self, max_num_seqs, tensor_parallel_size) -> None:
        # vLLMでモデルを初期化 tensor_parallel_sizeは使用するGPU数
        self.model = LLM(
            model="cyberagent/calm3-22b-chat",
            #model="cyberagent/calm2-7b-chat", # テスト用
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,  # バッチサイズに合わせて調整
            max_num_batched_tokens=16384,  # トークン数を増やす
            #max_model_len=1024, # テスト用
            download_dir="../cache",
        )

        self.sampling_params = SamplingParams(
            temperature=0.7, top_p=0.95, max_tokens=1024
        )

    def translate_batch(self, texts):
        prompts = [PROMPT.format(text=text) for text in texts]
        outputs = self.model.generate(prompts, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

    def translate_dataset(self, dataset, batch_size, num_samples, output_file):
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        translated_data = []

        for i in tqdm(range(0, len(dataset), batch_size), desc="バッチ処理中"):
            batch = dataset[i : i + batch_size]

            # データセットの構造に基づいて、適切なキーを使用
            en_texts = batch["en"]
            ja_texts = batch["ja"]

            translated_texts = self.translate_batch(en_texts)  # 英語から日本語に翻訳

            for en, ja, translated in zip(en_texts, ja_texts, translated_texts):
                translated_item = {
                    "en": en,
                    "ja_original": ja,
                    "ja_calm3": translated,
                }
                translated_data.append(translated_item)

            # 一定間隔で中間結果を保存
            if (i + batch_size) % (batch_size * 10) == 0:
                self.save_results(translated_data, output_file)

        # 最終結果を保存
        self.save_results(translated_data, output_file)
        print(f"翻訳が完了し、結果を'{output_file}' に保存しました。")

    def save_results(self, data, filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate a dataset using calm3 model"
    )
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to load")
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Number of samples to process"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="translated_dataset.json",
        help="File to save the translated results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for translation"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="set num gpu"
    )

    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name)

    translater = Translater(args.batch_size, args.tensor_parallel_size)

    translater.translate_dataset(
        dataset,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        output_file=args.output_file,
    )
