#!/bin/bash

# 自分のジョブID一覧を取得
job_ids=$(squeue | grep hrk | awk '{print $1}')

# 各ジョブに対してscontrolでNodeListとStdOutを表示
for job_id in $job_ids
do
    echo "Job ID: $job_id"
    scontrol show job $job_id | awk '
    /NodeList/ {print "NodeList: " $0}
    /StdOut/ {print "StdOut: " $0}
    /Reason/ {print "Reason: " $0}
    '
    echo "---------------------------------"
done

