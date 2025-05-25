# 基於Spark的聯邦學習

本項目實現了兩種聯邦學習方法的比較研究：
- **傳統聯邦學習**（Traditional FL）
- **基於Spark的聯邦學習**（Spark FL）

## 目錄

- [基於Spark的聯邦學習](#基於spark的聯邦學習)
  - [目錄](#目錄)
  - [項目結構](#項目結構)
  - [環境要求](#環境要求)
  - [快速開始](#快速開始)
    - [環境清理](#環境清理)
    - [啟動服務](#啟動服務)
  - [架構比較](#架構比較)
    - [傳統聯邦學習](#傳統聯邦學習)
    - [基於Spark的聯邦學習](#基於spark的聯邦學習-1)
    - [共同特性](#共同特性)
  - [性能評估](#性能評估)
    - [訓練結果摘要（20輪）](#訓練結果摘要20輪)
    - [詳細訓練數據](#詳細訓練數據)
    - [Spark FL 性能優勢因素](#spark-fl-性能優勢因素)
  - [容錯實驗](#容錯實驗)
    - [實驗1：數據分片貢獻失敗](#實驗1數據分片貢獻失敗)
    - [實驗2：Worker節點故障容錯實驗](#實驗2worker節點故障容錯實驗)
    - [實驗3：數據集更換實驗 - CIFAR-10](#實驗3數據集更換實驗---cifar-10)
  - [實驗設計原則](#實驗設計原則)
    - [數據影響平等原則](#數據影響平等原則)
    - [標準化測量](#標準化測量)
    - [監控方式](#監控方式)
  - [執行提示](#執行提示)
    - [權限設置](#權限設置)
    - [數據準備](#數據準備)

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

### 實驗2：Worker節點故障容錯實驗

**目標**：比較RDD血統追蹤機制與傳統FL checkpoint機制在worker節點故障時的容錯能力

**核心設計理念**：
- **RDD的真正優勢**：自動血統追蹤（lineage）可以在worker節點故障時自動重計算失敗的partition
- **傳統FL的劣勢**：worker節點故障時需要手動重啟和checkpoint恢復

**實驗架構設計**：
- **傳統FL架構**：1個Server + 4個Participant
  - 每個participant處理15,000個樣本（總共60,000樣本）
  - 故障場景：2個participant同時失效（50%數據分片失效）
  
- **Spark FL架構**：1個Master + 2個Worker Node
  - 每個worker node處理2個partition，共30,000個樣本
  - 故障場景：1個worker node失效（50%數據分片失效）

**模擬場景**：
- **Spark FL**：在第8輪模擬一個Worker節點故障（`docker stop exp2-spark-worker-1`）
  - 測試RDD自動檢測失敗partition並重新調度到其他worker
  - 驗證血統追蹤的自動重計算能力
  - 無需人工干預，系統自動恢復
  
- **傳統FL**：在第8輪模擬兩個Participant節點故障（`docker stop exp2-fl-participant-1 exp2-fl-participant-2`）
  - Server等待超時檢測participant離線
  - 需要手動重啟participant容器
  - 依賴checkpoint恢復訓練狀態

**技術實現重點**：

1. **RDD血統追蹤容錯**：
   ```python
   # Spark會自動檢測failed task並重新調度
   # RDD的lineage信息：
   # data_rdd -> map(load_shard) -> map(local_train) -> collect()
   # 任何stage失敗都會從血統重新計算
   ```

2. **故障注入策略**：
   - **Spark FL**：第8輪開始時殺死worker容器，讓Spark自動處理
   - **傳統FL**：第8輪開始時殺死2個participant容器，觀察server反應

**數據分片策略**：
| 方法 | 總樣本 | 分片數 | 每分片樣本數 | 故障影響 |
|------|-------|--------|-------------|---------|
| 傳統FL | 60,000 | 4個participant | 15,000 | 2個participant失效 = 30,000樣本丟失 |
| Spark FL | 60,000 | 2個worker × 2個partition | 15,000/partition | 1個worker失效 = 30,000樣本丟失 |

**評估指標**：
- **故障檢測時間**：
  - RDD：自動task失敗檢測（秒級）
  - 傳統FL：timeout檢測（分鐘級）

- **恢復時間**：
  - RDD：自動重新調度並重計算（秒級）
  - 傳統FL：手動重啟+checkpoint載入（分鐘級）

- **數據一致性**：
  - RDD：血統保證計算結果完全一致
  - 傳統FL：取決於checkpoint保存時機

- **人工干預需求**：
  - RDD：零干預（系統自動處理）
  - 傳統FL：需要手動操作容器重啟

### 實驗3：數據集更換實驗 - CIFAR-10

**目標**：評估更大數據集（CIFAR-10）對Spark RDD容錯機制與傳統FL容錯機制性能差異的影響

**數據集特性對比**：
| 特性 | MNIST | CIFAR-10 | 影響 |
|------|-------|----------|------|
| 圖像尺寸 | 28×28×1 | 32×32×3 | 數據量增加3.7倍 |
| 樣本數量 | 60,000 | 60,000 | 相同 |
| 複雜度 | 簡單手寫數字 | 彩色自然圖像 | 計算複雜度大幅增加 |
| 內存需求 | 低 | 高 | 內存壓力測試 |
| 訓練時間 | 短 | 長 | 容錯恢復開銷放大 |

**實驗設置**：
1. 保持相同的系統架構（2個數據分片）
2. 使用CIFAR-10數據集替代MNIST
3. 每個分片處理30,000個CIFAR-10樣本
4. 模型架構適配CIFAR-10（輸入通道改為3，增加網絡深度）

**實驗步驟**：
1. 準備CIFAR-10數據集
   ```bash
   # 下載和預處理CIFAR-10
   python prepare_cifar10.py
   ```

2. 執行基準訓練（20輪）
   ```bash
   # 傳統FL with CIFAR-10
   docker compose -f docker-compose-cifar10.yml up -d fl-server fl-participant-{1..2}
   
   # Spark FL with CIFAR-10
   docker compose -f docker-compose-cifar10.yml up -d spark-master spark-worker-1
   docker exec -it spark-master bash -c "cd /app && python spark_code/main_cifar10.py"
   ```

3. 重複實驗1：數據分片貢獻失敗測試
   - **第5輪故障模擬**：模擬一個數據分片不可用
   - **測試重點**：更大數據量下的容錯機制效率

**評估指標**：
- **性能影響對比**：
  - 訓練時間變化（MNIST vs CIFAR-10）
  - 內存使用峰值
  - 容錯恢復時間增長幅度

- **容錯機制壓力測試**：
  - Spark RDD血統追蹤在大數據量下的效率
  - 傳統FL checkpoint機制的I/O開銷
  - 故障恢復過程中的內存穩定性

- **系統韌性評估**：
  - 模型收斂穩定性（複雜數據集下）
  - 故障場景下的準確率保持能力
  - 網絡通信壓力下的容錯表現

**預期發現**：
1. **Spark RDD優勢放大**：
   - 更大數據集下，血統追蹤的精確恢復優勢更明顯
   - 分區級容錯避免大量數據重傳
   - 內存管理更高效

2. **傳統FL壓力暴露**：
   - Checkpoint I/O開銷在大數據集下顯著增加
   - Worker級容錯導致更多數據丟失
   - 內存壓力下故障恢復更困難

**實驗價值**：
此實驗將驗證Spark RDD容錯機制在**真實生產環境**（大數據集、高計算複雜度）下相比傳統FL容錯的優勢是否依然顯著，為大規模聯邦學習系統的容錯架構選擇提供決策依據。

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
