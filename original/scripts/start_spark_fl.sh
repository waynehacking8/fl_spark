#!/bin/bash

# 設置 Spark 環境變量
export SPARK_HOME=/opt/spark
export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
export PATH=$SPARK_HOME/bin:$PATH

# 創建數據目錄
mkdir -p data/mnist

# 安裝依賴
pip install -r requirements.txt

# 啟動 Spark 聯邦學習
python spark_code/main.py

# 等待完成
wait 