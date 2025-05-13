#!/bin/bash

# 故障模擬腳本

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印帶顏色的消息
print_message() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# 模擬節點遺失
simulate_node_failure() {
    local service_name=$1
    local delay=$2
    
    print_message "模擬節點遺失：將在 ${delay} 秒後停止 ${service_name}"
    sleep $delay
    
    print_message "停止 ${service_name}..."
    docker compose stop $service_name
    
    print_message "${service_name} 已停止"
}

# 模擬資料遺失
simulate_data_loss() {
    local service_name=$1
    local delay=$2
    
    print_message "模擬資料遺失：將在 ${delay} 秒後刪除 ${service_name} 的資料"
    sleep $delay
    
    print_message "刪除 ${service_name} 的資料..."
    docker compose exec $service_name rm -rf /app/data/*
    
    print_message "${service_name} 的資料已刪除"
}

# 模擬資料損毀
simulate_data_corruption() {
    local service_name=$1
    local delay=$2
    
    print_message "模擬資料損毀：將在 ${delay} 秒後損毀 ${service_name} 的資料"
    sleep $delay
    
    print_message "損毀 ${service_name} 的資料..."
    docker compose exec $service_name dd if=/dev/urandom of=/app/data/corrupted bs=1M count=10
    
    print_message "${service_name} 的資料已損毀"
}

# 主函數
main() {
    local test_type=$1
    local service_name=$2
    local delay=$3
    
    case $test_type in
        "node")
            simulate_node_failure $service_name $delay
            ;;
        "data")
            simulate_data_loss $service_name $delay
            ;;
        "corruption")
            simulate_data_corruption $service_name $delay
            ;;
        *)
            print_error "未知的測試類型：$test_type"
            print_error "用法：$0 [node|data|corruption] <service_name> <delay_seconds>"
            exit 1
            ;;
    esac
}

# 執行主函數
main "$@" 