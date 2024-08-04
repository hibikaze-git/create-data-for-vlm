# re_translate
- 既存の英語データセットをcalm3で翻訳

## データの翻訳
```
python calm3_batch_cc-by-sa.py team-hatakeyama-phase2/commoncatalog-cc-by-sa-ja --batch_size 1000 --tensor_parallel_size 1

python calm3_batch_cc-by-sa-complex.py team-hatakeyama-phase2/commoncatalog-cc-by-sa-ja-complex --batch_size 1000 --tensor_parallel_size 1
```
