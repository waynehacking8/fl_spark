#!/bin/bash

echo "=== 開始CIFAR-10完整實驗（20輪，batch_size=32，ResNet模型） ==="

# 清理之前的進程
pkill -f "participant_cifar10.py" 2>/dev/null || true
pkill -f "server_cifar10.py" 2>/dev/null || true
sleep 2

echo "=== 開始傳統FL (20輪) ==="
echo "🔄 正常模式：無故障注入"

# 啟動服務器（後台）
cd traditional_code
python server_cifar10.py --rounds 20 --model resnet --mode normal &
SERVER_PID=$!
sleep 3

# 啟動參與者
python participant_cifar10.py --participant_id 1 --rounds 20 &
PARTICIPANT1_PID=$!

python participant_cifar10.py --participant_id 2 --rounds 20 &
PARTICIPANT2_PID=$!

# 等待完成
wait $SERVER_PID
wait $PARTICIPANT1_PID 2>/dev/null || true
wait $PARTICIPANT2_PID 2>/dev/null || true

echo "Traditional FL完成!"
cd ..

echo "=== 開始 Spark FL (20輪) ==="
echo "🔄 正常模式：無故障注入"

# 運行Spark FL
cd spark_code
python spark_fl_cifar10.py --rounds 20 --model resnet --mode normal
cd ..

echo "=== 實驗結果對比 ==="
echo "Traditional FL結果:"
cat results/traditional/normal/cifar10_normal_results.csv

echo ""
echo "Spark FL結果:"
cat results/spark/normal/cifar10_spark_normal_results.csv

echo "🎉 CIFAR-10 完整實驗完成!" 