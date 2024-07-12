import json
import os

import datasets
from tqdm import tqdm
from vllm import LLM, SamplingParams

# vLLMでモデルを初期化 tensor_parallel_sizeは使用するGPU数
model = LLM(
    model="cyberagent/calm3-22b-chat",
    tensor_parallel_size=8,
    max_num_seqs=1000,  # バッチサイズに合わせて調整
    max_num_batched_tokens=16384,  # トークン数を増やす
    # quantization="awq",  # または "squeezellm", "gptq", "awq" など
    # dtype="bfloat16",  # または "float16", "bfloat16", "half"
    # block_size=32,
)
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)

# データセットの読み込み
dataset = datasets.load_dataset("atsushi3110/en-ja-parallel-corpus-augmented")["train"]

# データセットの構造を確認
print("Dataset structure:")
print(dataset[0])
print("Dataset columns:", dataset.column_names)


def translate_batch(texts):
    prompts = [
        f"""<|im_start|>system
あなたは親切なAIアシスタントです。<|im_end|>
<|im_start|>user
以下を日本語に翻訳してください。
{text}<|im_end|>
<|im_start|>assistant"""
        for text in texts
    ]
    outputs = model.generate(prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


def translate_dataset(
    dataset, batch_size=32, num_samples=None, output_file="test.json"
):
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    translated_data = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="バッチ処理中"):
        batch = dataset[i : i + batch_size]

        # データセットの構造に基づいて、適切なキーを使用
        en_texts = batch["en"]  # 'en'キーが存在すると仮定
        ja_texts = batch["ja"]  # 'ja'キーが存在すると仮定

        translated_texts = translate_batch(en_texts)  # 英語から日本語に翻訳

        for en, ja, translated in zip(en_texts, ja_texts, translated_texts):
            translated_item = {
                "en": en,
                "ja_original": ja,
                "ja_calm3": translated,
            }
            translated_data.append(translated_item)

        # 一定間隔で中間結果を保存
        if (i + batch_size) % (batch_size * 10) == 0:
            save_results(translated_data, output_file)

    # 最終結果を保存
    save_results(translated_data, output_file)
    print(f"翻訳が完了し、結果を'{output_file}' に保存しました。")


def save_results(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    batch_size = 1000  # バッチサイズを設定
    num_samples = 1000  # 処理するサンプル数を設定（Noneでデータセット全体を処理）
    output_file = "en_ja_parallel_calm3.json"

    # 既存のファイルがある場合、バックアップを作成
    if os.path.exists(output_file):
        backup_file = f"{output_file}.bak"
        os.rename(output_file, backup_file)
        print(f"既存のファイルを {backup_file} にバックアップしました。")

    translate_dataset(
        dataset, batch_size=batch_size, num_samples=num_samples, output_file=output_file
    )
