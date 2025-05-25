#!/bin/bash
cd "$(dirname "$0")"

echo "=== 🚀 同時啟動聯邦學習故障容錯實驗 ==="
echo "🔥 解決第一輪聚合延遲問題"
echo "✅ 自動編號修正機制已啟用"
echo

# 殺死舊進程
echo "清理舊進程..."
pkill -f "python.*participant_fixed.py" || true
pkill -f "python.*server_fixed.py" || true
sleep 3

# 清理舊結果
echo "清理舊結果..."
mkdir -p ../results/traditional/checkpoints
rm -f ../results/traditional/checkpoints/results.csv
rm -f ../results/traditional/checkpoints/traditional_fl_accuracy.csv
rm -f ../results/traditional/checkpoints/model_round_*.pth
sleep 1

echo "🚀 同時啟動服務器和參與者..."

# 同時啟動服務器和參與者（最小延遲）
{
    echo "啟動服務器..."
    nohup python server_fixed.py > server_sync.log 2>&1 &
    SERVER_PID=$!
    echo "服務器PID: $SERVER_PID"
    
    # 很短的延遲讓服務器初始化
    sleep 2
    
    echo "同時啟動所有參與者..."
    # 使用後台作業同時啟動所有參與者
    nohup python participant_fixed.py 1 > participant_sync_1.log 2>&1 &
    P1_PID=$!
    nohup python participant_fixed.py 2 > participant_sync_2.log 2>&1 &
    P2_PID=$!
    nohup python participant_fixed.py 3 > participant_sync_3.log 2>&1 &
    P3_PID=$!
    nohup python participant_fixed.py 4 > participant_sync_4.log 2>&1 &
    P4_PID=$!
    
    echo "參與者PID: $P1_PID, $P2_PID, $P3_PID, $P4_PID"
} &

wait

echo
echo "🎯 同時啟動實驗完成！"
echo "📊 修復功能："
echo "   ✅ 第8輪故障偵測與恢復"
echo "   ✅ 輪次驗證機制"  
echo "   ✅ 自動編號修正（7,8,9,10,11...連續）"
echo "   ✅ 減少第一輪聚合延遲"
echo
echo "📋 監控指令:"
echo "   觀察服務器: tail -f server_sync.log"
echo "   觀察參與者1: tail -f participant_sync_1.log"
echo "   觀察參與者2: tail -f participant_sync_2.log"
echo "   觀察參與者3: tail -f participant_sync_3.log"
echo "   觀察參與者4: tail -f participant_sync_4.log"
echo "   查看結果: cat ../results/traditional/checkpoints/results.csv"
echo
echo "🔧 技術改進："
echo "   1. 同時啟動減少第一輪延遲"
echo "   2. 自動編號修正機制"
echo "   3. 故障偵測與恢復"
echo "   4. 輪次驗證與去重"
echo

# 實時監控第一輪
echo "⏰ 監控第一輪啟動速度..."
start_time=$(date +%s)

for i in {1..30}; do
    sleep 2
    if [ -f "server_sync.log" ]; then
        # 檢查第一輪是否開始
        if grep -q "開始第 1 輪" server_sync.log; then
            first_round_time=$(date +%s)
            echo "📍 第1輪已開始 (啟動後 $((first_round_time - start_time)) 秒)"
            
            # 檢查第一個參與者連接
            if grep -q "參與者.*第1輪連接" server_sync.log; then
                first_connect_time=$(date +%s)
                echo "📍 第一個參與者已連接 (啟動後 $((first_connect_time - start_time)) 秒)"
                break
            fi
        fi
    fi
    echo "⏳ 等待第1輪開始... ($i/30)"
done

echo
echo "✅ 同時啟動實驗監控完成"
echo "📈 第一輪延遲已最小化"
echo "🎯 實驗正在進行中..." 