#!/bin/bash

# CIFAR-10聯邦學習實驗運行腳本
# 支持normal和exp1兩種實驗模式

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 功能函數
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 檢查依賴
check_dependencies() {
    log_info "檢查依賴項..."
    
    # 檢查Python環境
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未找到，請安裝Python3"
        exit 1
    fi
    
    # 檢查必要的Python包
    local packages=("torch" "torchvision" "pyspark" "numpy")
    for package in "${packages[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            log_error "Python包 $package 未找到，請安裝"
            exit 1
        fi
    done
    
    log_success "依賴檢查完成"
}

# 準備數據
prepare_data() {
    log_info "準備CIFAR-10數據..."
    
    cd ../
    if [ ! -f "data/cifar10_train_part1.pt" ] || [ ! -f "data/cifar10_train_part2.pt" ]; then
        log_info "數據文件不存在，正在下載和準備..."
        python3 prepare_cifar10.py
        
        if [ $? -eq 0 ]; then
            log_success "數據準備完成"
        else
            log_error "數據準備失敗"
            exit 1
        fi
    else
        log_success "數據文件已存在"
    fi
    cd scripts/
}

# 運行Traditional FL
run_traditional_fl() {
    local mode=$1
    log_info "運行Traditional FL (${mode}模式)..."
    
    cd ../traditional_code/
    
    # 創建結果目錄
    mkdir -p "../results/traditional/${mode}"
    
    # 啟動服務器
    log_info "啟動FL服務器..."
    python3 server_cifar10.py --mode ${mode} --rounds 20 --model simple > "../results/traditional/${mode}/server_${mode}.log" 2>&1 &
    SERVER_PID=$!
    sleep 3
    
    # 檢查服務器是否成功啟動
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        log_error "服務器啟動失敗"
        return 1
    fi
    
    # 啟動參與者
    log_info "啟動參與者..."
    python3 participant_cifar10.py 1 --mode ${mode} --rounds 20 --model simple > "../results/traditional/${mode}/participant1_${mode}.log" 2>&1 &
    PARTICIPANT1_PID=$!
    
    python3 participant_cifar10.py 2 --mode ${mode} --rounds 20 --model simple > "../results/traditional/${mode}/participant2_${mode}.log" 2>&1 &
    PARTICIPANT2_PID=$!
    
    # 等待訓練完成
    wait $SERVER_PID
    wait $PARTICIPANT1_PID 2>/dev/null || true
    wait $PARTICIPANT2_PID 2>/dev/null || true
    
    log_success "Traditional FL (${mode}模式) 完成"
    cd ../scripts/
}

# 運行Spark FL
run_spark_fl() {
    local mode=$1
    log_info "運行Spark FL (${mode}模式)..."
    
    cd ../spark_code/
    
    # 創建結果目錄
    mkdir -p "../results/spark/${mode}"
    
    # 運行Spark FL
    python3 spark_fl_cifar10.py --mode ${mode} --rounds 20 --model simple --partitions 2 > "../results/spark/${mode}/spark_${mode}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        log_success "Spark FL (${mode}模式) 完成"
    else
        log_error "Spark FL (${mode}模式) 失敗"
        return 1
    fi
    
    cd ../scripts/
}

# 清理進程
cleanup() {
    log_info "清理進程..."
    
    # 殺死所有相關進程
    pkill -f "server_cifar10.py" 2>/dev/null || true
    pkill -f "participant_cifar10.py" 2>/dev/null || true
    pkill -f "spark_fl_cifar10.py" 2>/dev/null || true
    
    sleep 2
    log_success "清理完成"
}

# 分析結果
analyze_results() {
    local mode=$1
    log_info "分析 ${mode} 模式結果..."
    
    # 檢查結果文件
    local traditional_file="../results/traditional/${mode}/cifar10_${mode}_results.csv"
    local spark_file="../results/spark/${mode}/cifar10_spark_${mode}_results.csv"
    
    if [ -f "$traditional_file" ] && [ -f "$spark_file" ]; then
        log_success "${mode} 模式結果文件已生成:"
        echo "  Traditional FL: $traditional_file"
        echo "  Spark FL: $spark_file"
        
        # 簡單對比
        local traditional_acc=$(tail -1 "$traditional_file" | cut -d',' -f3)
        local spark_acc=$(tail -1 "$spark_file" | cut -d',' -f3)
        
        echo "最終準確率對比 (${mode}模式):"
        echo "  Traditional FL: ${traditional_acc}%"
        echo "  Spark FL: ${spark_acc}%"
    else
        log_warning "${mode} 模式部分結果文件缺失"
    fi
}

# 顯示使用方法
show_usage() {
    echo "用法: $0 [mode] [options]"
    echo ""
    echo "模式:"
    echo "  normal    - 正常聯邦學習（無故障注入）"
    echo "  exp1      - 實驗1：數據分片貢獻失敗（第5輪）"
    echo "  both      - 運行兩種模式"
    echo ""
    echo "選項:"
    echo "  --traditional-only  - 只運行Traditional FL"
    echo "  --spark-only       - 只運行Spark FL"
    echo "  --no-cleanup       - 不執行清理"
    echo "  --help             - 顯示此幫助信息"
    echo ""
    echo "示例:"
    echo "  $0 normal              # 運行normal模式"
    echo "  $0 exp1               # 運行exp1模式"
    echo "  $0 both               # 運行兩種模式"
    echo "  $0 normal --spark-only # 只運行normal模式的Spark FL"
}

# 主函數
main() {
    local mode=""
    local run_traditional=true
    local run_spark=true
    local do_cleanup=true
    
    # 解析參數
    while [[ $# -gt 0 ]]; do
        case $1 in
            normal|exp1|both)
                mode="$1"
                shift
                ;;
            --traditional-only)
                run_spark=false
                shift
                ;;
            --spark-only)
                run_traditional=false
                shift
                ;;
            --no-cleanup)
                do_cleanup=false
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "未知參數: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # 檢查模式參數
    if [ -z "$mode" ]; then
        log_error "請指定實驗模式"
        show_usage
        exit 1
    fi
    
    # 設置錯誤時清理
    trap cleanup EXIT
    
    # 檢查依賴
    check_dependencies
    
    # 準備數據
    prepare_data
    
    # 運行實驗
    if [ "$mode" = "both" ]; then
        modes=("normal" "exp1")
    else
        modes=("$mode")
    fi
    
    for current_mode in "${modes[@]}"; do
        log_info "=== 開始 ${current_mode} 模式實驗 ==="
        
        if [ "$run_traditional" = true ]; then
            run_traditional_fl "$current_mode"
            sleep 2
        fi
        
        if [ "$run_spark" = true ]; then
            run_spark_fl "$current_mode"
            sleep 2
        fi
        
        analyze_results "$current_mode"
        
        log_success "=== ${current_mode} 模式實驗完成 ==="
        echo ""
    done
    
    # 清理
    if [ "$do_cleanup" = true ]; then
        cleanup
    fi
    
    log_success "🎉 CIFAR-10實驗全部完成！"
}

# 執行主函數
main "$@" 