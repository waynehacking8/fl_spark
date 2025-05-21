# 基於Spark的聯邦學習

本項目實現了兩種聯邦學習方法的比較研究：
- **傳統聯邦學習**（Traditional FL）
- **基於Spark的聯邦學習**（Spark FL）

## 目錄

- [項目結構](#項目結構)
- [環境要求](#環境要求)
- [快速開始](#快速開始)
- [架構比較](#架構比較)
- [性能評估](#性能評估)
- [容錯實驗](#容錯實驗)
- [執行提示](#執行提示)

## 項目結構

```
.
├── traditional_code/     # 傳統聯邦學習代碼
├── spark_code/          # 基於Spark的聯邦學習代碼
├── data/               # 數據集
├── evaluation/         # 評估腳本
├── results/           # 結果輸出
└── docker-compose.yml # Docker配置
```

## 環境要求

- Docker
- Docker Compose
- NVIDIA GPU（用於加速訓練）

## 快速開始

### 環境清理

```bash
# 移除所有未使用的映像、卷、構建緩存
cd original
docker compose down
docker system prune -af --volumes
docker builder prune -af

# 或完全清理所有Docker資源
docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q) && docker rmi $(docker images -q) -f
```

### 啟動服務

1. **傳統聯邦學習**
   ```bash
   docker compose run --rm data-init
   docker compose up -d fl-server fl-participant-{1..2}
   ```

2. **基於Spark的聯邦學習**
   ```bash
   docker compose up -d spark-master spark-worker-1
   docker exec -it original-spark-master-1 bash -c "cd /app && python /app/spark_code/main.py"
   ```

3. **使用現有映像重新啟動**
   ```bash
   docker compose -f docker-compose.yml up -d fl-server fl-participant-1 fl-participant-2
   ```

## 架構比較

### 傳統聯邦學習
- 使用Socket通信
- 2個獨立參與者節點
- 每個節點處理30,000個樣本
- 手動管理節點生命週期

### 基於Spark的聯邦學習
- 使用Spark RDD進行數據分發
- 2個並行分區
- 每個分區處理30,000個樣本
- 自動資源分配管理

### 共同特性
- 數據集：MNIST（60,000個訓練樣本）
- 模型：CNN架構
- 批次大小：32
- 本地訓練輪數：5
- 優化器：SGD
- 學習率：0.01

## 性能評估

### 訓練結果摘要（20輪）

| 指標 | 傳統FL | Spark FL | 改進比例 |
|-----|--------|---------|---------|
| 總訓練時間 | 664.82秒 | 199.95秒 | 快3.3倍 |
| 最終準確率 | 98.88% | 98.89% | 相當 |

### 詳細訓練數據

| Round | Trad. Time (s) | Spark Time (s) | Trad. Acc (%) | Spark Acc (%) |
|------:|--------------:|---------------:|--------------:|--------------:|
| 1 | 36.73 | 12.17 | 97.20 | 96.40 |
| 5 | 165.71 | 52.16 | 98.64 | 98.66 |
| 10 | 334.56 | 101.87 | 98.83 | 98.76 |
| 15 | 499.85 | 150.95 | 98.88 | 98.78 |
| 20 | 664.82 | 199.95 | 98.88 | 98.89 |

> 完整數據表格請參見原始研究報告

### Spark FL 性能優勢因素

1. **數據分片優化**：減少分片數量（從16個到2個），降低數據傳輸和聚合開銷
2. **GPU利用率提升**：更有效地利用GPU資源，避免CPU計算瓶頸
3. **通信效率**：減少節點間通信次數，大幅降低通信開銷
4. **資源分配**：Spark的自動資源分配機制更高效
5. **模型架構統一**：兩種方法使用相同的CNN架構，確保公平比較

## 容錯實驗

### 實驗1：數據分片貢獻失敗

**目標**：比較當某輪中無法獲取一個數據分片的處理結果時的系統反應

**模擬場景**：
- **傳統FL**：在第R輪中，模擬參與者節點離線（`docker stop fl-participant-1`）
- **Spark FL**：在第R輪中，模擬數據文件不可讀（`chmod 000 <分片文件路徑>`）

**評估指標**：
- 故障檢測能力
- 聚合機制處理策略
- 自動恢復能力
- 對模型性能的影響

### 實驗2：中央協調節點故障

**目標**：比較中央控制節點故障時的系統行為和恢復能力

**模擬場景**：
- **傳統FL**：模擬服務器故障（`docker stop fl-server`）
- **Spark FL**：模擬Spark主節點故障（`docker stop spark-master`）

**評估指標**：
- 系統即時響應
- 訓練狀態保存
- 恢復流程效率
- 狀態丟失風險

### 實驗3：節點故障與恢復測試

**目標**：測試系統在節點故障後的自動恢復能力

**實驗步驟**：
1. 開始正常訓練過程
2. 在第5輪中停止一個處理節點
   - 傳統FL：`docker stop fl-participant-1`
   - Spark FL：`docker stop spark-worker-1`
3. 等待3分鐘，然後重啟節點
4. 觀察系統是否自動恢復訓練

**評估指標**：
- 自動檢測恢復能力
- 訓練連續性
- 恢復後性能表現

## 實驗設計原則

### 數據影響平等原則
- 確保兩種方法在每個實驗中失敗的數據量相同
- 每個節點故障實驗影響1個分片（30,000個樣本）

### 標準化測量
- 統一指標：訓練時間、準確率曲線、收斂輪數
- 每個實驗重複3次取平均值，減少隨機因素

### 監控方式
- 傳統FL：Docker日誌
- Spark FL：Spark UI

## 執行提示

### 權限設置
如果結果目錄由root創建，需設置正確權限：
```bash
sudo chown -R 1001:1001 results data
```
其中`1001`是bitnami基礎映像的默認非root用戶。

### 數據準備
- `data-init`服務會運行`traditional_code/prepare_mnist.py`下載並分割數據
- 首次運行後，後續啟動不需重新處理數據，減少I/O開銷