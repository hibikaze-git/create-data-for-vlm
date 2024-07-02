# complex_reasoning
- phi3-visionによる合成・phi3による翻訳を行い、データを合成
- 作業は基本的にこのディレクトリで行う
- [こちら](https://huggingface.co/datasets/common-canvas/commoncatalog-cc-by-sa/tree/main)のデータセットの4の一部を対象に合成を行いました。

# 使い方
## データのダウンロード
```
python partial_downloader.py 4
```

## データの合成
```
python synthesis_data.py ./commoncatalog-cc-by-sa-download/4/least_dim_range=1024-2048 ./images/4/least_dim_range=1024-2048 ./jsonl/4/least_dim_range=1024-2048
```
