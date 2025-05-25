#!/bin/bash
# 
# CIFAR-10 Federated Learning Fault Tolerance Experiment
# 實驗三：數據集更換實驗 - CIFAR-10
# 

set -e

echo "🎯 CIFAR-10 聯邦學習故障容錯實驗"
echo "=============================================="

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 實驗參數
PARTICIPANTS=2
ROUNDS=20
MODEL_TYPE="simple"  # 使用簡化模型提高速度
BATCH_SIZE=64
LEARNING_RATE=0.001
LOCAL_EPOCHS=5

# 目錄設置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXP_DIR"

echo -e "${BLUE}實驗目錄: $EXP_DIR${NC}"
echo -e "${BLUE}參數配置:${NC}"
echo "  - 參與者數量: $PARTICIPANTS"
echo "  - 訓練輪數: $ROUNDS"
echo "  - 模型類型: $MODEL_TYPE"
echo "  - 批次大小: $BATCH_SIZE"
echo "  - 學習率: $LEARNING_RATE"
echo "  - 本地輪數: $LOCAL_EPOCHS"

# 函數：清理進程
cleanup() {
    echo -e "\n${YELLOW}正在清理進程...${NC}"
    pkill -f "server_cifar10.py" 2>/dev/null || true
    pkill -f "participant_cifar10.py" 2>/dev/null || true
    pkill -f "spark_fl_cifar10.py" 2>/dev/null || true
    sleep 2
}

# 函數：檢查Python依賴
check_dependencies() {
    echo -e "\n${BLUE}[1/7] 檢查Python依賴...${NC}"
    
    python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')" || {
        echo -e "${RED}❌ PyTorch未安裝${NC}"
        exit 1
    }
    
    python3 -c "import torchvision; print(f'TorchVision版本: {torchvision.__version__}')" || {
        echo -e "${RED}❌ TorchVision未安裝${NC}" 
        exit 1
    }
    
    python3 -c "import pyspark; print(f'PySpark版本: {pyspark.__version__}')" || {
        echo -e "${RED}❌ PySpark未安裝${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✅ 依賴檢查通過${NC}"
}

# 函數：數據準備
prepare_data() {
    echo -e "\n${BLUE}[2/7] 準備CIFAR-10數據...${NC}"
    
    if [[ -f "data/cifar10_train_part1.pt" && -f "data/cifar10_train_part2.pt" && -f "data/cifar10_test.pt" ]]; then
        echo -e "${GREEN}✅ CIFAR-10數據已存在，跳過下載${NC}"
    else
        echo "正在下載和準備CIFAR-10數據..."
        python3 prepare_cifar10.py || {
            echo -e "${RED}❌ 數據準備失敗${NC}"
            exit 1
        }
        echo -e "${GREEN}✅ CIFAR-10數據準備完成${NC}"
    fi
}

# 函數：測試模型架構
test_models() {
    echo -e "\n${BLUE}[3/7] 測試模型架構...${NC}"
    python3 models.py || {
        echo -e "${RED}❌ 模型測試失敗${NC}"
        exit 1
    }
    echo -e "${GREEN}✅ 模型架構測試通過${NC}"
}

# 函數：運行傳統FL實驗
run_traditional_fl() {
    echo -e "\n${BLUE}[4/7] 運行傳統FL實驗...${NC}"
    
    # 清理之前的結果
    rm -f results/traditional/cifar10_results.csv
    rm -f results/traditional/model_round_*.pth
    
    # 啟動服務器
    echo "正在啟動傳統FL服務器..."
    python3 traditional_code/server_cifar10.py \
        --participants $PARTICIPANTS \
        --rounds $ROUNDS \
        --model $MODEL_TYPE \
        --timeout 30 &
    
    SERVER_PID=$!
    echo "服務器PID: $SERVER_PID"
    sleep 5
    
    # 啟動參與者
    echo "正在啟動參與者..."
    for i in $(seq 1 $PARTICIPANTS); do
        python3 traditional_code/participant_cifar10.py $i \
            --model $MODEL_TYPE \
            --batch_size $BATCH_SIZE \
            --lr $LEARNING_RATE \
            --epochs $LOCAL_EPOCHS \
            --rounds $ROUNDS &
        
        PARTICIPANT_PID=$!
        echo "參與者 $i PID: $PARTICIPANT_PID"
        sleep 2
    done
    
    # 等待服務器完成
    echo "等待傳統FL實驗完成..."
    wait $SERVER_PID
    
    # 清理參與者進程
    cleanup
    
    if [[ -f "results/traditional/cifar10_results.csv" ]]; then
        echo -e "${GREEN}✅ 傳統FL實驗完成${NC}"
        tail -5 results/traditional/cifar10_results.csv
    else
        echo -e "${RED}❌ 傳統FL實驗失敗${NC}"
        exit 1
    fi
}

# 函數：運行Spark FL實驗
run_spark_fl() {
    echo -e "\n${BLUE}[5/7] 運行Spark FL實驗...${NC}"
    
    # 清理之前的結果
    rm -f results/spark/cifar10_spark_results.csv
    rm -f results/spark/spark_model_round_*.pth
    
    # 運行Spark FL
    echo "正在運行Spark FL..."
    python3 spark_code/spark_fl_cifar10.py \
        --participants $PARTICIPANTS \
        --rounds $ROUNDS \
        --epochs $LOCAL_EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LEARNING_RATE \
        --model $MODEL_TYPE || {
        echo -e "${RED}❌ Spark FL實驗失敗${NC}"
        exit 1
    }
    
    if [[ -f "results/spark/cifar10_spark_results.csv" ]]; then
        echo -e "${GREEN}✅ Spark FL實驗完成${NC}"
        tail -5 results/spark/cifar10_spark_results.csv
    else
        echo -e "${RED}❌ Spark FL實驗失敗${NC}"
        exit 1
    fi
}

# 函數：生成分析報告
generate_analysis() {
    echo -e "\n${BLUE}[6/7] 生成分析報告...${NC}"
    
    # 檢查結果文件
    if [[ -f "results/traditional/cifar10_results.csv" && -f "results/spark/cifar10_spark_results.csv" ]]; then
        echo "正在生成比較分析..."
        
        # 創建簡單的比較報告
        cat > results/cifar10_experiment_summary.txt << EOF
CIFAR-10 聯邦學習故障容錯實驗結果摘要
================================

實驗配置:
- 數據集: CIFAR-10 (50,000訓練樣本 + 10,000測試樣本)
- 參與者數量: $PARTICIPANTS
- 訓練輪數: $ROUNDS
- 模型類型: $MODEL_TYPE
- 故障注入: 第8輪參與者1故障

傳統FL結果:
$(tail -1 results/traditional/cifar10_results.csv)

Spark FL結果:
$(tail -1 results/spark/cifar10_spark_results.csv)

生成時間: $(date)
EOF
        
        echo -e "${GREEN}✅ 分析報告已生成: results/cifar10_experiment_summary.txt${NC}"
    else
        echo -e "${RED}❌ 結果文件不完整，無法生成分析報告${NC}"
    fi
}

# 函數：顯示實驗結果
show_results() {
    echo -e "\n${BLUE}[7/7] 實驗結果總結${NC}"
    echo "=============================================="
    
    if [[ -f "results/cifar10_experiment_summary.txt" ]]; then
        cat results/cifar10_experiment_summary.txt
    fi
    
    echo -e "\n${GREEN}🎉 CIFAR-10實驗完成！${NC}"
    echo -e "${BLUE}結果文件位置:${NC}"
    echo "  - 傳統FL: results/traditional/cifar10_results.csv"
    echo "  - Spark FL: results/spark/cifar10_spark_results.csv"
    echo "  - 摘要報告: results/cifar10_experiment_summary.txt"
    
    # 顯示文件大小對比
    echo -e "\n${BLUE}文件大小對比:${NC}"
    ls -lh results/traditional/cifar10_results.csv 2>/dev/null || echo "  傳統FL結果文件不存在"
    ls -lh results/spark/cifar10_spark_results.csv 2>/dev/null || echo "  Spark FL結果文件不存在"
}

# 主執行流程
main() {
    # 設置陷阱處理中斷
    trap cleanup EXIT INT TERM
    
    echo -e "${YELLOW}開始CIFAR-10聯邦學習故障容錯實驗...${NC}"
    
    check_dependencies
    prepare_data
    test_models
    run_traditional_fl
    run_spark_fl
    generate_analysis
    show_results
    
    echo -e "\n${GREEN}✅ 所有實驗步驟完成！${NC}"
}

# 處理命令行參數
case "${1:-all}" in
    "deps")
        check_dependencies
        ;;
    "data")
        prepare_data
        ;;
    "test")
        test_models
        ;;
    "traditional")
        run_traditional_fl
        ;;
    "spark")
        run_spark_fl
        ;;
    "analysis")
        generate_analysis
        ;;
    "all")
        main
        ;;
    *)
        echo "用法: $0 [deps|data|test|traditional|spark|analysis|all]"
        echo "  deps       - 檢查依賴"
        echo "  data       - 準備數據"
        echo "  test       - 測試模型"
        echo "  traditional - 運行傳統FL"
        echo "  spark      - 運行Spark FL"
        echo "  analysis   - 生成分析"
        echo "  all        - 運行完整實驗"
        exit 1
        ;;
esac 