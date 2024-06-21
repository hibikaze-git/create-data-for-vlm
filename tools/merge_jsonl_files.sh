#!/bin/bash

# 引数が指定されているか確認
# bash merge_jsonl_files.sh ./extract merged.jsonl
if [ $# -ne 2 ]; then
  echo "Usage: $0 <directory> <output_file>"
  exit 1
fi

# 指定されたディレクトリが存在するか確認
if [ ! -d "$1" ]; then
  echo "Directory not found: $1"
  exit 1
fi

# 出力ファイルを初期化
output_file="$2"
> "$output_file"

# 指定ディレクトリ以下の全.jsonlファイルを1つのファイルにマージ
find "$1" -type f -name "*.jsonl" | while read -r file
do
  echo "$file"
  cat "$file" >> "$output_file"
done

echo "Merged all .jsonl files into: $output_file"
