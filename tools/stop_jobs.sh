#!/bin/bash

# 引数のチェック
if [ "$#" -ne 1 ]; then
    echo "使用法: $0 <ノード名>"
    exit 1
fi

# 引数を変数に格納
node_name=$1

# 自分のジョブID一覧を取得
job_ids=$(squeue | grep vlm_synt | grep $node_name | awk '{print $1}')

# 各ジョブに対してscontrolでNodeListとStdOutを表示
for job_id in $job_ids
do
    echo "Stop Job ID: $job_id"
    scancel $job_id
done

