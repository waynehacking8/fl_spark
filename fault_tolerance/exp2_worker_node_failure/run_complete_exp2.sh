#!/bin/bash

echo "=========================================="
echo "EXP2: 完整的 Worker 節點故障容錯實驗"
echo "Traditional FL vs Spark FL 對比"
echo "=========================================="

# 設置顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 檢查是否在正確的目錄
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}[ERROR]${NC} 請在 exp2_worker_node_failure 目錄下運行此腳本"
    exit 1
fi

echo -e "${BLUE}[INFO]${NC} 開始完整的 EXP2 實驗流程..."

# 步驟1：運行 Traditional FL
echo -e "\n${PURPLE}=== 步驟 1: Traditional FL 容錯實驗 ===${NC}"
if [ ! -f "simple_run/results/results.csv" ]; then
    echo -e "${BLUE}[INFO]${NC} 運行 Traditional FL 實驗..."
    cd simple_run
    ./run_simultaneous_experiment.sh
    cd ..
    
    if [ -f "simple_run/results/results.csv" ]; then
        echo -e "${GREEN}[SUCCESS]${NC} Traditional FL 實驗完成"
    else
        echo -e "${RED}[ERROR]${NC} Traditional FL 實驗失敗"
    fi
else
    echo -e "${GREEN}[SKIP]${NC} Traditional FL 結果已存在"
fi

# 步驟2：運行 Spark FL
echo -e "\n${PURPLE}=== 步驟 2: Spark FL 容錯實驗 ===${NC}"
if [ ! -f "results/spark/results.csv" ]; then
    echo -e "${BLUE}[INFO]${NC} 運行 Spark FL 實驗..."
    ./run_spark_fl.sh
    
    if [ -f "results/spark/results.csv" ]; then
        echo -e "${GREEN}[SUCCESS]${NC} Spark FL 實驗完成"
    else
        echo -e "${RED}[ERROR]${NC} Spark FL 實驗失敗"
    fi
else
    echo -e "${GREEN}[SKIP]${NC} Spark FL 結果已存在"
fi

# 步驟3：結果分析
echo -e "\n${PURPLE}=== 步驟 3: 結果分析與對比 ===${NC}"

echo -e "${BLUE}[INFO]${NC} 分析 Spark FL 結果..."
python analyze_spark_results.py

echo -e "${BLUE}[INFO]${NC} 生成對比分析..."
python compare_fl_methods.py

# 步驟4：生成完整報告
echo -e "\n${PURPLE}=== 步驟 4: 實驗報告生成 ===${NC}"

# 檢查所有結果文件
echo -e "${BLUE}[INFO]${NC} 檢查實驗結果文件..."

files_to_check=(
    "simple_run/results/results.csv:Traditional FL 結果"
    "simple_run/results/performance.png:Traditional FL 性能圖"
    "results/spark/results.csv:Spark FL 結果"
    "results/spark/performance.png:Spark FL 性能圖"
    "results/spark/spark_fault_tolerance_analysis.png:Spark FL 分析圖"
    "results/exp2_fl_comparison.png:對比圖表"
    "results/exp2_comparison_report.md:對比報告"
)

for file_desc in "${files_to_check[@]}"; do
    IFS=':' read -r file desc <<< "$file_desc"
    if [ -f "$file" ]; then
        echo -e "  ✅ $desc: $file"
    else
        echo -e "  ❌ $desc: $file"
    fi
done

# 生成實驗摘要
echo -e "\n${PURPLE}=== 實驗摘要 ===${NC}"

if [ -f "simple_run/results/results.csv" ] && [ -f "results/spark/results.csv" ]; then
    echo -e "${GREEN}[COMPLETE]${NC} 兩種方法的實驗都已完成"
    
    # 提取關鍵指標
    trad_rounds=$(tail -n +2 simple_run/results/results.csv | wc -l)
    trad_final_acc=$(tail -1 simple_run/results/results.csv | cut -d',' -f3)
    trad_total_time=$(tail -1 simple_run/results/results.csv | cut -d',' -f2)
    
    spark_rounds=$(tail -n +2 results/spark/results.csv | wc -l)
    spark_final_acc=$(tail -1 results/spark/results.csv | cut -d',' -f3)
    spark_total_time=$(tail -1 results/spark/results.csv | cut -d',' -f2)
    
    echo -e "\n📊 ${BLUE}實驗結果對比${NC}:"
    echo -e "   Traditional FL: $trad_rounds 輪, 準確率 $trad_final_acc%, 總時間 ${trad_total_time}s"
    echo -e "   Spark FL:       $spark_rounds 輪, 準確率 $spark_final_acc%, 總時間 ${spark_total_time}s"
    
elif [ -f "simple_run/results/results.csv" ]; then
    echo -e "${YELLOW}[PARTIAL]${NC} 僅完成 Traditional FL 實驗"
elif [ -f "results/spark/results.csv" ]; then
    echo -e "${YELLOW}[PARTIAL]${NC} 僅完成 Spark FL 實驗"
else
    echo -e "${RED}[INCOMPLETE]${NC} 實驗未完成"
fi

echo -e "\n${GREEN}[FINISH]${NC} EXP2 完整實驗流程結束"
echo -e "${BLUE}[INFO]${NC} 查看詳細結果:"
echo -e "   - Traditional FL: simple_run/results/"
echo -e "   - Spark FL:       results/spark/"
echo -e "   - 對比分析:       results/" 