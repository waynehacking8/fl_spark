#!/bin/bash

echo "=== CIFAR-10 防過擬合 20輪對比實驗 ==="

# 清理進程
pkill -f "participant_cifar10.py" 2>/dev/null || true
pkill -f "server_cifar10.py" 2>/dev/null || true
sleep 3

echo "🔄 第一階段：傳統FL 20輪實驗"
echo "================================"

cd traditional_code

# 啟動服務器
echo "啟動傳統FL服務器..."
python server_cifar10.py --rounds 20 --model resnet --mode normal &
SERVER_PID=$!

# 等待服務器啟動
sleep 5

# 啟動參與者
echo "啟動參與者..."
python participant_cifar10.py --participant_id 1 --model resnet --rounds 20 --mode normal &
PARTICIPANT1_PID=$!

sleep 2
python participant_cifar10.py --participant_id 2 --model resnet --rounds 20 --mode normal &
PARTICIPANT2_PID=$!

# 等待完成
echo "等待傳統FL實驗完成..."
wait $SERVER_PID
echo "傳統FL服務器完成!"

# 清理參與者進程
kill $PARTICIPANT1_PID 2>/dev/null || true
kill $PARTICIPANT2_PID 2>/dev/null || true

sleep 5

echo ""
echo "🚀 第二階段：Spark FL 20輪實驗"
echo "================================"

cd ../spark_code

# 運行Spark FL
echo "啟動Spark FL..."
python spark_fl_cifar10.py --rounds 20 --model resnet --mode normal

echo ""
echo "🎉 兩個版本20輪實驗完成！"
echo "================================"

echo "📊 結果對比："
echo ""
echo "傳統FL結果："
tail -5 ../results/traditional/normal/cifar10_normal_results.csv

echo ""
echo "Spark FL結果："
tail -5 ../results/spark/normal/cifar10_spark_normal_results.csv

echo ""
echo "實驗完成！結果文件位置："
echo "- 傳統FL: results/traditional/normal/cifar10_normal_results.csv"
echo "- Spark FL: results/spark/normal/cifar10_spark_normal_results.csv" 