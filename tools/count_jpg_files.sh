#!/bin/bash

# 引数が指定されているか確認
# bash count_image_files.sh ./images
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

# .pngファイルの数をカウント
png_count=$(find "$1" -type f -name "*.png" | wc -l)

# .jpgと.pngファイルの合計数を計算
total_count=$((jpg_count + png_count))

# 結果を表示
echo "Total .jpg files in directory $1: $jpg_count"
echo "Total .png files in directory $1: $png_count"
echo "Total image files (.jpg + .png) in directory $1: $total_count"
