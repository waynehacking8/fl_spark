#!/bin/bash

echo "=========================================="
echo "Traditional FL EXP2 - Optimized Version"  
echo "No fixed delay, 30s timeout consistency"
echo "=========================================="

# 清理環境
echo "[1/6] Cleaning environment..."
pkill -f "python.*server_fixed.py" 2>/dev/null || true
pkill -f "python.*participant_fixed.py" 2>/dev/null || true
sleep 2

# 創建必要的目錄
mkdir -p results/traditional/checkpoints
mkdir -p results/traditional/plots

# 清理舊結果
echo "[2/6] Cleaning old results..."
rm -f results/traditional/checkpoints/results.csv
rm -f results/traditional/checkpoints/traditional_fl_accuracy.csv
rm -f results/traditional/checkpoints/model_round_*.pth
rm -f results/traditional/plots/*.png

# 準備數據
echo "[3/6] Preparing data..."
if [ ! -f "data/mnist_test.pt" ] || [ ! -f "data/participant_1_shard.pt" ]; then
    python3 prepare_data.py
    echo "Data preparation completed"
else
    echo "Data already exists, skipping preparation"
fi

# 啟動服務器
echo "[4/6] Starting server..."
python3 server_fixed.py &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"
sleep 3

# 啟動4個參與者（背景運行）
echo "[5/6] Starting 4 participants..."
for i in {1..4}; do
    echo "Starting participant $i..."
    python3 participant_fixed.py $i &
    PARTICIPANT_PIDS[$i]=$!
    echo "Participant $i started with PID: ${PARTICIPANT_PIDS[$i]}"
    sleep 1
done

echo "All participants started. Waiting for experiment completion..."

# 等待服務器完成（服務器會在20輪完成後自動退出）
wait $SERVER_PID
echo "Server completed!"

# 確保所有參與者進程結束
echo "Cleaning up participant processes..."
for i in {1..4}; do
    if kill -0 ${PARTICIPANT_PIDS[$i]} 2>/dev/null; then
        kill ${PARTICIPANT_PIDS[$i]} 2>/dev/null || true
    fi
done

sleep 2

# 生成可視化結果
echo "[6/6] Generating visualizations..."
if [ -f "results/traditional/checkpoints/results.csv" ]; then
    python3 plot_traditional_results_en.py
    echo "Visualizations generated!"
else
    echo "❌ No results file found! Experiment may have failed."
    exit 1
fi

echo "=========================================="
echo "Traditional FL EXP2 Completed!"
echo "Results:"
echo "  📊 CSV: results/traditional/checkpoints/results.csv"
echo "  📈 Plots: results/traditional/plots/"
echo "==========================================" 