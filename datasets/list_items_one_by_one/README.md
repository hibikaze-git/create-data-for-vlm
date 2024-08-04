# list_items_one_by_one
- MS-COCO 2017に対して、Florence-2-largeを用いてデータの合成を行う

## データのダウンロード
```
bash download_mscoco.sh
```

## データの合成
```
python synthesis_data.py ./train2017 ./images ./jsonl
```
