#!/bin/bash

# 引数が指定されているか確認
# bash count_jsonl_lines.sh ./extract
if [ $# -ne 1 ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# 指定されたディレクトリが存在するか確認
if [ ! -d "$1" ]; then
  echo "Directory not found: $1"
  exit 1
fi

# 合計行数を初期化
total_lines=0

# 指定ディレクトリ以下の全.jsonlファイルの行数をカウント
for file in $(find "$1" -type f -name "*.jsonl")
do
  line_count=$(wc -l < "$file")
  #echo "$file: $line_count lines"
  total_lines=$((total_lines + line_count))
done

# 合計行数を表示
echo "Total lines in all .jsonl files: $total_lines"
