# EXP2 Spark FL Worker 節點故障容錯實驗

## 🚀 快速開始

### 僅運行 Spark FL 實驗
```bash
./run_spark_fl.sh
```

### 完整對比實驗 (Traditional FL + Spark FL)
```bash
./run_complete_exp2.sh
```

## 🏗️ Spark FL 架構

```
Spark Master (Driver)
├── Worker 1 (處理分區 0, 1)
└── Worker 2 (處理分區 2, 3)

故障場景：第8輪時分區 0, 1 故障
- 模擬 participant 1&2 離線
- RDD 血統追蹤自動重計算失敗分區
- 無需人工干預
```

## 📊 實驗重點

### 故障注入機制
- **第8輪**：分區 0 和 1 模擬故障
- **故障類型**：Worker 節點完全無響應 (60秒延遲)
- **預期行為**：Spark 自動檢測並重新調度失敗任務

### RDD 容錯優勢
1. **血統追蹤**：自動記錄數據變換過程
2. **分區恢復**：只重計算失敗的分區
3. **零干預**：無需手動重啟節點
4. **秒級檢測**：快速識別任務失敗

## 📁 結果文件

執行完成後檢查：
- `results/spark/results.csv` - 訓練結果
- `results/spark/performance.png` - 性能圖表
- `results/spark/spark_fault_tolerance_analysis.png` - 故障分析圖
- `results/exp2_fl_comparison.png` - 對比圖表 (如果運行完整實驗)

## 🔧 手動執行步驟

1. **準備數據**
   ```bash
   docker compose run --rm data-init
   ```

2. **啟動 Spark 集群**
   ```bash
   docker compose up -d spark-master spark-worker-1 spark-worker-2
   ```

3. **運行實驗**
   ```bash
   docker exec -it exp2-spark-master bash -c "cd /app && python /app/main.py"
   ```

4. **分析結果**
   ```bash
   python analyze_spark_results.py
   ```

## 🎯 預期結果

- **總輪數**：20 輪
- **故障輪次**：第8輪有明顯延遲
- **最終準確率**：~98-99%
- **容錯驗證**：第8輪後繼續正常訓練

## 🆚 與 Traditional FL 對比

| 特性 | Traditional FL | Spark FL |
|------|----------------|----------|
| 故障檢測 | 60秒超時 | 秒級自動檢測 |
| 恢復機制 | Checkpoint | RDD血統追蹤 |
| 人工干預 | 需要重啟節點 | 完全自動 |
| 容錯粒度 | 節點級 | 分區級 |

---

**技術棧**：Apache Spark, PySpark, PyTorch, Docker  
**容錯機制**：RDD血統追蹤 + 自動任務重新調度 