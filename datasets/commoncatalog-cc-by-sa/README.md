# commoncatalog-cc-by-sa
- phi3-visionによる合成・phi3による翻訳を行い、データを合成
- 作業は基本的にこのディレクトリで行う

# 使い方
## データのダウンロード
```
python partial_downloader.py 0
```

## データの合成
```
python synthesis_data.py ./commoncatalog-cc-by-sa-download/1/least_dim_range=1024-2048 ./images/1/least_dim_range=1024-2048 ./jsonl/1/least_dim_range=1024-2048 --num_processes 3
```
