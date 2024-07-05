#!/bin/bash

# 引数が指定されているか確認
if [ $# -ne 2 ]; then
  echo "Usage: $0 <node_list> <GPU数>"
  exit 1
fi

# ノードリストを引数として取得
node_list=$1
gpu_num=$2

# 無限ループでノードの存在をチェック
while true
do
  # ReqNodeNotAvailで保留中のジョブIDを取得
  job_ids=$(squeue -h -t PENDING -o "%i %r %j" | grep "0705cc_" | grep "ReqNodeNotAvail, May be reserved for other job" | awk '{print $1}')

  # ジョブIDが存在しない場合の処理
  if [ -z "$job_ids" ]; then
    echo "No jobs found with reason: ReqNodeNotAvail, May be reserved for other job"
    exit 0
  fi

  nodes_exist=false

  # 各ジョブIDに対してReqNodeListを取得
  for job_id in $job_ids
  do
    job_info=$(scontrol show job $job_id)
    req_node_list=$(echo "$job_info" | grep -oP 'ReqNodeList=\K[^\s]+')

    # 引数で指定したノードリストに存在するか確認
    if [[ $node_list == *"$req_node_list" ]]; then
      echo "JobId: $job_id, ReqNodeList: $req_node_list"
      nodes_exist=true
      break
    fi
  done

  # ノードの存在状態を表示
  if $nodes_exist; then
    bash tools/update_0705cc_gpu_num.sh ${node_list} stop ${gpu_num}
  else
    echo "ノードが存在しません"
    exit 1
  fi

  # 5秒待機してから再チェック
  sleep 5
done

