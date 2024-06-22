#!/bin/bash

# 引数が指定されているか確認
# bash count_jsonl_files.sh ./extract
if [ $# -ne 1 ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# 指定されたディレクトリが存在するか確認
if [ ! -d "$1" ]; then
  echo "Directory not found: $1"
  exit 1
fi

# .jsonlファイルの個数をカウント
file_count=$(find "$1" -type f -name "*.jsonl" | wc -l)

# ファイルの個数を表示
echo "Total number of .jsonl files: $file_count"
