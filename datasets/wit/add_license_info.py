import json
import urllib.parse

import pandas as pd
import requests
from tqdm import tqdm


def get_license(image_name_list):
    # APIエンドポイント
    url = f"https://commons.wikimedia.org/w/api.php"

    image_name_list = [f"File:{image_name}" for image_name in image_name_list]

    image_names = "|".join(image_name_list)

    # パラメータ
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": image_names,
        "iiprop": "url|user|extmetadata",
    }

    # リクエスト送信
    response = requests.get(url, params=params)
    data = response.json()

    print(data)

    # ライセンス情報の取得
    pages = data.get("query", {}).get("pages", {})

    license_info_list = []

    for page_id, page_info in pages.items():
        license_info = "No License Info"

        if "imageinfo" in page_info:
            imageinfo = page_info.get("imageinfo", [])[0]
            extmetadata = imageinfo.get("extmetadata", {})
            license_info = extmetadata.get("LicenseShortName", {}).get(
                "value", "No License Info"
            )

        license_info_list.append(license_info)

    return license_info_list


filename = "wit_v1.train.all-1percent_sample"

# TSVファイルを読み込む
file_path = f"data/{filename}.tsv"
df = pd.read_csv(file_path, sep="\t")

# 日本語の行を抽出
df_ja = df[df["language"] == "ja"]

# JSONL形式で保存
output_file_path = f"output/{filename}_with_license.jsonl"

# 既に存在する場合は処理済みのURLを取得
processed_urls = set()
try:
    with open(output_file_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            processed_urls.add(record["image_url"])
except FileNotFoundError:
    pass

# tqdmを使ってプログレスバーを表示しながらライセンス情報を取得し、1行ずつ保存
with open(output_file_path, "a", encoding="utf-8") as f:
    data_batch = []
    image_name_batch = []

    for i, row in tqdm(df_ja.iterrows(), total=df_ja.shape[0]):
        if row["image_url"] in processed_urls:
            continue

        data_batch.append(row)

        image_name = urllib.parse.unquote(row["image_url"].split("/")[-1])

        image_name_batch.append(image_name)

        if len(image_name_batch) == 50:
            license_info_list = get_license(image_name_batch)

            for data, license_info in zip(data_batch, license_info_list):
                data["license"] = license_info
                f.write(json.dumps(data.to_dict(), ensure_ascii=False) + "\n")
                processed_urls.add(row["image_url"])

            data_batch = []
            image_name_batch = []
