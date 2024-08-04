# diverse_questions
- 画像キャプションと質問集に基づき、画像に対するQAを合成
- 質問集は [MM-Instruct](https://arxiv.org/abs/2406.19736)のカテゴリを参考に作成しました

## データの合成
```
python calm3_batch.py alfredplpl/commoncatalog-cc-by-recap --batch_size 1000 --tensor_parallel_size 1
```
