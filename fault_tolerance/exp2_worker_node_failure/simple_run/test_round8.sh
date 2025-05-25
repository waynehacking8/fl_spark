#!/bin/bash

echo "=========================================="
echo "Testing Round 8 Fault Tolerance"  
echo "=========================================="

# 清理環境
pkill -f "python.*server_fixed.py" 2>/dev/null || true
pkill -f "python.*participant_fixed.py" 2>/dev/null || true
sleep 2

# 創建測試版本的服務器，只運行9輪
cp server_fixed.py server_test.py
sed -i 's/NUM_ROUNDS = 20/NUM_ROUNDS = 9/' server_test.py

# 清理舊結果
rm -f results/traditional/checkpoints/results.csv
rm -f results/traditional/checkpoints/model_round_*.pth

# 準備數據
if [ ! -f "data/mnist_test.pt" ]; then
    python3 prepare_data.py
fi

# 啟動服務器
echo "Starting test server (9 rounds only)..."
python3 server_test.py &
SERVER_PID=$!
sleep 3

# 啟動4個參與者
echo "Starting 4 participants..."
for i in {1..4}; do
    python3 participant_fixed.py $i &
    sleep 1
done

# 等待完成
wait $SERVER_PID

echo "Test completed. Checking results..."
if [ -f "results/traditional/checkpoints/results.csv" ]; then
    echo "Results:"
    cat results/traditional/checkpoints/results.csv
else
    echo "No results found!"
fi

# 清理
pkill -f "python.*participant_fixed.py" 2>/dev/null || true
rm -f server_test.py 