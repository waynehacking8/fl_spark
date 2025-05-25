#!/bin/bash

echo "=== 測試防過擬合ResNet (10輪) ==="

# 清理進程
pkill -f "participant_cifar10.py" 2>/dev/null || true
pkill -f "server_cifar10.py" 2>/dev/null || true
sleep 2

cd traditional_code

# 啟動服務器
echo "啟動防過擬合服務器..."
python server_cifar10.py --rounds 10 --model resnet --mode normal &
SERVER_PID=$!

# 等待服務器啟動
sleep 5

# 啟動參與者
echo "啟動參與者..."
python participant_cifar10.py --participant_id 1 --model resnet --rounds 10 &
PARTICIPANT1_PID=$!

sleep 2
python participant_cifar10.py --participant_id 2 --model resnet --rounds 10 &
PARTICIPANT2_PID=$!

# 等待完成
echo "等待實驗完成..."
wait $SERVER_PID
echo "服務器完成!"

# 清理參與者進程
kill $PARTICIPANT1_PID 2>/dev/null || true
kill $PARTICIPANT2_PID 2>/dev/null || true

echo "=== 結果分析 ==="
echo "檢查損失是否穩定下降..."
tail -10 ../results/traditional/normal/cifar10_normal_results.csv

echo "測試完成!" 