# complex_reasoning
- phi3-visionによる合成・phi3による翻訳を行い、データを合成
- [こちら](https://huggingface.co/datasets/common-canvas/commoncatalog-cc-by-sa/tree/main)のデータセットの4の一部を対象に合成を行いました。
- phi3よりcyberagent/calm3-22b-chatの方が翻訳性能が高く、再翻訳を行いました。../re_translateを参照してください

# 使い方
## データのダウンロード
commoncatalog-cc-by-sa-download配下に画像がダウンロードされます。
```
python partial_downloader.py 4
```

## データの合成
- 以下は一例です
- 複数のleast_dim_rangeに対して合成を行っています。./sbatch内のスクリプトを参照してください
```
python synthesis_data.py ./commoncatalog-cc-by-sa-download/4/least_dim_range=1024-2048 ./images/4/least_dim_range=1024-2048 ./jsonl/4/least_dim_range=1024-2048
```
