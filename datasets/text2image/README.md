# text2image
- テキストデータ→html→スクリーンショットでOCRデータの自動生成を試みる

# 使い方
## テキスト→html
```
python gen_html.py stockmark/ner-wikipedia-dataset
```

## html→image
### インストール
```
cd datasets/text2image
git clone https://github.com/vgalin/html2image.git
cd ../../
docker compose --profile html2image build
docker compose --profile html2image up -d
docker compose exec html2image bash
pip install datasets Pillow
apt install -y fonts-ipafont fonts-ipaexfont
```
### 実行
```
python gen_image.py --num_workers 8
```