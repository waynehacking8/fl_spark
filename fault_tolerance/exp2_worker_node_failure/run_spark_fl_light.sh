#!/bin/bash

echo "=========================================="
echo "EXP2: Spark FL 輕量化實驗"
echo "=========================================="

# 設置顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}[INFO]${NC} 清理舊容器..."
docker compose down > /dev/null 2>&1

echo -e "${BLUE}[INFO]${NC} 確保數據已準備..."
if [ ! -f "data/mnist_train_part1.pt" ]; then
    echo -e "${YELLOW}[PREPARE]${NC} 準備數據分片..."
    docker compose run --rm data-init
else
    echo -e "${GREEN}[SKIP]${NC} 數據分片已存在"
fi

echo -e "${BLUE}[INFO]${NC} 僅啟動 Spark 集群（不啟動 FL services）..."
docker compose up -d spark-master spark-worker-1 spark-worker-2

echo -e "${BLUE}[INFO]${NC} 等待 Spark 集群啟動..."
sleep 8

echo -e "${GREEN}[START]${NC} 開始 Spark FL 訓練..."
echo -e "${YELLOW}[注意]${NC} 第8輪將模擬 Worker 節點故障"

# 執行 Spark FL 訓練
docker exec exp2-spark-master bash -c "cd /app && python /app/main.py"

echo -e "${GREEN}[DONE]${NC} Spark FL 訓練完成"

echo -e "${BLUE}[INFO]${NC} 檢查結果..."
if [ -f "results/spark/results.csv" ]; then
    echo -e "${GREEN}[SUCCESS]${NC} 結果文件已生成"
    echo "最後5行結果："
    tail -5 results/spark/results.csv
else
    echo -e "${RED}[ERROR]${NC} 未找到結果文件"
fi

echo -e "${BLUE}[INFO]${NC} 停止容器以釋放資源..."
docker compose down

echo -e "${GREEN}[COMPLETE]${NC} EXP2 實驗完成" 