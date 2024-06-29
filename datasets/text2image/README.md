# text2image
- テキストデータ→html→スクリーンショットでOCRデータを自動生成を試みる

# 使い方
## テキスト→html
```
python gen_html.py stockmark/ner-wikipedia-dataset
```

## html→image
インストール
```
cd datasets/text2image
git clone https://github.com/vgalin/html2image.git
docker compose --profile html2image build
docker compose --profile html2image up -d
docker compose exec html2image bash
pip install datasets
apt install -y fonts-ipafont fonts-ipaexfont
```
