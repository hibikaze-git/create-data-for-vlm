#!/bin/bash

# 引数のチェック
if [ "$#" -ne 2 ]; then
    echo "使用法: $0 <ノード名> <GPU数>"
    exit 1
fi

# 引数を変数に格納
node_name=$1
gpu_num=$2

while true; do
    # `squeue` の出力を変数に格納
    squeue_output=$(squeue)

    # 条件をチェック
    if echo "$squeue_output" | grep -q "0705cc_.*ext_kan_.*ReqNodeNotAvail.*UnavailableNodes:${node_name}"; then
        # 条件に一致した場合にスクリプトを実行
        bash tools/update_0705cc_gpu_num.sh ${node_name} stop ${gpu_num}
    else
        # 条件に一致しない場合はループを終了
        echo "条件に一致しません。スクリプトを終了します。"
        break
    fi

    # 10秒待機
    sleep 10
done

