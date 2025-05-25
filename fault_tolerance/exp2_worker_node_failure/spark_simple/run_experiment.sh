#!/bin/bash

echo "=========================================="
echo "Spark FL 簡化版 Worker Node Fault Tolerance 實驗"
echo "=========================================="

# 設置環境變量
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "[1/3] 檢查 Python 環境..."
python3 --version

echo "[2/3] 準備數據..."
python3 prepare_data.py

echo "[3/3] 開始 Spark FL 實驗..."
python3 spark_fl_simple.py

echo "=========================================="
echo "實驗完成！"
echo "結果已保存到 results/ 目錄"
echo "==========================================" 