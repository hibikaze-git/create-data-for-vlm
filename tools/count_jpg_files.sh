#!/bin/bash

# 引数が指定されているか確認
# bash count_jpg_files.sh ./images
if [ $# -ne 1 ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# 指定されたディレクトリが存在するか確認
if [ ! -d "$1" ]; then
  echo "Directory not found: $1"
  exit 1
fi

# .jpgファイルの数をカウント
jpg_count=$(find "$1" -type f -name "*.jpg" | wc -l)

# 結果を表示
echo "Total .jpg files in directory $1: $jpg_count"
