# cauldron
- HuggingFaceM4/the_cauldron の一部のデータを日本語に翻訳
- 翻訳数は最大50kに制限しているので注意
- 学習には未使用

# 使い方
## the_cauldronへのidの付与
- the_cauldronデータセットには一意なidがなく、処理が中断された場合等に途中から再開するのが難しいです
- 翻訳前の準備として、以下のnotebookを使用してidを付与し、自信のHFリポジトリにuploadします
```
../../notebooks/cauldron-add-id.ipynb
../../notebooks/cauldron-add-id-plotqa.ipynb
```

## データの翻訳
plotqa以外
```
python translate.py vqarad
```
plotqas
```
python translate_plotqa_shuffle.py plotqa_shuffle 0-6250
```
