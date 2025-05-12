#!/bin/bash

# 設置 CUDA 相關環境變量
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# 設置 Spark 環境變量
export SPARK_HOME=/opt/spark
export PATH=$PATH:$SPARK_HOME/bin

# 根據節點類型啟動相應的服務
if [ "$SPARK_MODE" == "master" ]; then
    # 啟動 Spark Master
    $SPARK_HOME/sbin/start-master.sh -h 0.0.0.0
    # 保持容器運行
    tail -f $SPARK_HOME/logs/*
else
    # 啟動 Spark Worker
    $SPARK_HOME/sbin/start-worker.sh "spark://$SPARK_MASTER_HOST:7077" \
        -h "$SPARK_LOCAL_HOST" \
        -m "$SPARK_WORKER_MEMORY" \
        -c "$SPARK_WORKER_CORES"
    # 保持容器運行
    tail -f $SPARK_HOME/logs/*
fi