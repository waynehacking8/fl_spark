#!/bin/bash

echo "=== CIFAR-10 實驗1：數據分片貢獻失敗 ==="
echo "模擬第5輪參與者1離線的情況"
echo "=========================================="

# 清理進程
pkill -f "participant_cifar10.py" 2>/dev/null || true
pkill -f "server_cifar10.py" 2>/dev/null || true
sleep 3

echo "🧪 第一階段：傳統FL exp1實驗"
echo "================================"

cd traditional_code

# 啟動服務器
echo "啟動傳統FL服務器 (exp1模式)..."
python server_cifar10.py --rounds 20 --model resnet --mode exp1 &
SERVER_PID=$!

# 等待服務器啟動
sleep 5

# 啟動參與者
echo "啟動參與者..."
python participant_cifar10.py --participant_id 1 --model resnet --rounds 20 --mode exp1 &
PARTICIPANT1_PID=$!

sleep 2
python participant_cifar10.py --participant_id 2 --model resnet --rounds 20 --mode exp1 &
PARTICIPANT2_PID=$!

# 等待完成
echo "等待傳統FL exp1實驗完成..."
wait $SERVER_PID
echo "傳統FL exp1服務器完成!"

# 清理參與者進程
kill $PARTICIPANT1_PID 2>/dev/null || true
kill $PARTICIPANT2_PID 2>/dev/null || true

sleep 5

echo ""
echo "🚀 第二階段：Spark FL exp1實驗"
echo "================================"

cd ../spark_code

# 運行Spark FL exp1
echo "啟動Spark FL exp1..."
python spark_fl_cifar10.py --rounds 20 --model resnet --mode exp1

echo ""
echo "🎉 exp1實驗完成！"
echo "================================"

echo "📊 結果對比："
echo ""
echo "傳統FL exp1結果："
tail -5 ../results/traditional/exp1/cifar10_exp1_results.csv

echo ""
echo "Spark FL exp1結果："
tail -5 ../results/spark/exp1/cifar10_spark_exp1_results.csv

echo ""
echo "實驗完成！結果文件位置："
echo "- 傳統FL exp1: results/traditional/exp1/cifar10_exp1_results.csv"
echo "- Spark FL exp1: results/spark/exp1/cifar10_spark_exp1_results.csv" 