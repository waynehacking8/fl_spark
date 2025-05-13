# Federated Learning with Spark

這個專案實現了兩種聯邦學習方法：
1. 傳統的聯邦學習（Traditional FL）
2. 基於 Spark 的聯邦學習（Spark FL）

## 專案結構

```
.
├── traditional_code/     # 傳統聯邦學習代碼
├── spark_code/          # Spark 聯邦學習代碼
├── data/               # 數據集
├── evaluation/         # 評估腳本
├── results/           # 結果輸出
└── docker-compose.yml # Docker 配置
```

## 環境要求

- Docker
- Docker Compose
- NVIDIA GPU (用於加速訓練)

## 快速開始

1. 啟動傳統聯邦學習：
```bash
docker compose up -d fl-server fl-participant-{1..16}
```

2. 啟動 Spark 聯邦學習：
```bash
docker compose up -d spark-master spark-worker-{1..2}
```

## 配置說明

- `docker-compose.yml` 包含了所有服務的配置
- 每個參與者（participant）分配了 3,750 個訓練樣本
- 使用 MNIST 數據集進行訓練
- 模型使用 CNN 架構

## 性能比較

### 架構比較
1. 傳統聯邦學習：
   - 使用 Socket 通信
   - 16 個獨立參與者節點
   - 每個節點處理 3,750 個樣本
   - 手動管理節點生命週期

2. Spark 聯邦學習：
   - 使用 Spark RDD 進行數據分發
   - 16 個並行分區
   - 每個分區處理 3,750 個樣本
   - 自動管理資源分配

### 數據吞吐量分析
兩種方法在相同條件下具有相同的數據處理吞吐量：

1. 數據量：
   - 總樣本數：60,000 個
   - 每個處理單元：3,750 個樣本
   - 並行度：16 個處理單元

2. 處理方式：
   - 相同的批次大小（32）
   - 相同的本地訓練輪數（5）
   - 相同的模型架構（CNN）
   - 相同的優化器（SGD）

3. 硬件資源：
   - 相同的 GPU（NVIDIA GeForce RTX 4070 Ti SUPER）
   - 相同的內存配置（4G 限制，2G 預留）

4. 聚合策略：
   - 兩種方法都使用聯邦平均
   - 每輪訓練後進行一次模型聚合

因此，雖然實現方式不同，但兩種方法在數據處理吞吐量上是完全相同的。

## 容錯性實驗

### 實驗目標
比較傳統 FL 和 Spark FL 在面對不同類型故障時的穩健性和恢復能力。

### 故障類型與模擬方法

1. 節點遺失（優先測試）
   - 模擬方法：使用 `docker stop` 命令停止隨機選擇的節點
   - 測試場景：
     * 單個節點故障
     * 多個節點同時故障
     * 不同時間點的故障（訓練開始後 30%、50%、70%）
   - 觀察指標：
     * 故障檢測時間
     * 自動恢復能力
     * 訓練完成時間
     * 最終模型準確率

2. Shard 遺失（進階測試）
   - 模擬方法：修改訓練代碼，隨機跳過部分本地訓練數據
   - 測試參數：
     * 數據丟失率：10%、20%、30%
     * 丟失模式：隨機、連續、週期性
   - 觀察指標：
     * 訓練穩定性
     * 模型收斂速度
     * 最終準確率

3. 數據掉包/損毀（進階測試）
   - 模擬方法：在模型更新傳輸前加入噪聲或隨機丟棄
   - 測試參數：
     * 噪聲強度：0.1、0.2、0.3
     * 丟包率：10%、20%、30%
   - 觀察指標：
     * 模型更新質量
     * 訓練穩定性
     * 最終準確率

### 實驗流程

1. 基準測試
   - 在無故障情況下運行兩個系統
   - 記錄每輪訓練的測試準確率
   - 建立基準性能曲線

2. 故障測試
   - 按優先順序進行三種故障測試
   - 每種故障類型進行多次重複測試
   - 記錄所有相關指標

3. 數據收集
   - 系統日誌
   - 訓練準確率曲線
   - 故障恢復時間
   - 資源利用率

4. 分析比較
   - 故障恢復能力
   - 訓練穩定性
   - 最終模型性能
   - 資源利用效率

### 預期結果

1. 節點遺失測試：
   - Spark FL 應該能自動檢測故障並重試任務
   - 傳統 FL 可能需要手動干預
   - 兩種方法在模型準確率上差異不大

2. Shard 遺失測試：
   - 兩種方法都能繼續訓練
   - 準確率可能略有下降
   - 收斂速度可能變慢

3. 數據掉包/損毀測試：
   - 兩種方法都能保持訓練
   - 準確率可能受到影響
   - 收斂可能不穩定

## 監控

- 傳統 FL：通過 Docker 日誌查看訓練進度
- Spark FL：通過 Spark UI 監控任務執行情況

## 實驗設計

### 核心設定

1. 數據集
   - 使用標準 MNIST 數據集
   - 初始採用 IID 數據分割
   - 未來可擴展到 Non-IID 情境

2. 模型架構
   - 使用相同的 PyTorch CNN 模型
   - 確保兩種方法使用相同架構
   - 固定模型參數和超參數

3. 系統實現
   - Spark FL：
     * Master/Worker 架構
     * 使用 Spark broadcast 分發模型
     * 使用 reduce/treeAggregate 聚合更新
   - 傳統 FL：
     * 客戶端-伺服器模式
     * TCP Socket 通信
     * 點對點模型更新傳輸

4. 實驗參數
   - 參與者數量：10、50、100
   - 固定聯邦學習總輪數
   - 固定本地訓練 Epoch 數
   - 使用標準 FedAvg 聚合策略

### 評估指標

1. 訓練效率
   - 端到端訓練時間
   - 故障恢復時間
   - 資源利用率（CPU/內存）

2. 通信開銷
   - 總數據傳輸量
   - 網絡帶寬使用率
   - 通信延遲

3. 模型性能
   - 每輪測試準確率
   - 最終模型準確率
   - 收斂速度

4. 容錯能力
   - 故障檢測時間
   - 恢復成功率
   - 訓練完成率

### 實驗環境

1. 容器化部署
   - Docker Compose 管理
   - 環境一致性保證
   - 依賴自動安裝

2. 監控工具
   - Docker stats
   - Spark UI
   - 自定義日誌

3. 數據收集
   - 實時日誌記錄
   - 性能指標採集
   - 結果文件導出

### 時程安排

1. 第一週：核心實現
   - Spark FL 基本流程
   - 數據分發機制
   - 容錯邏輯實現

2. 第二週：系統完善
   - 數據加載優化
   - 評估功能實現
   - 容錯測試

3. 第三週：實驗執行
   - 基準性能測試
   - 故障模擬實驗
   - 數據收集

4. 第四週：分析報告
   - 數據分析
   - 圖表繪製
   - 報告撰寫

### 成功指標

1. 效率提升
   - Spark FL 總訓練時間顯著優於傳統 FL
   - 考慮容錯開銷後仍保持優勢

2. 通信優化
   - 總通信開銷顯著降低
   - 網絡利用率提升

3. 性能可比
   - 無故障時準確率相近
   - 故障後仍能達到可接受水平

4. 容錯驗證
   - 成功完成預定訓練輪數
   - 故障恢復時間在可接受範圍

### 未來擴展

1. 功能擴展
   - Spark MLlib 深度集成
   - 動態參與者選擇
   - 差分隱私保護

2. 架構優化
   - 實時聯邦學習
   - 自適應參數調優
   - 監控系統整合

## 執行提示與性能觀察

> 以下內容來自 2025-05-12 的實測記錄，供後續使用者參考。

### 一、權限/I/O 設定
1. **結果目錄寫權限**
   Docker 映射的 `results/` 若由 root 建立，進容器會寫不進去，請在主機上執行：
   ```bash
   sudo chown -R 1001:1001 results data
   ```
   其中 `1001` 為 bitnami 基底映像的預設非 root 使用者。

2. **預先切分 MNIST shard**
   - `data-init` 服務已改為執行 `traditional_code/prepare_mnist.py`，只會在第一次啟動時下載 + 產生
     `mnist_train_part1.pt … mnist_train_part16.pt`（每檔 3 750 筆）。
   - 之後重跑傳統 FL 不再解壓 gzip，SSD I/O 峰值消失。

3. **一次性重建指令**  （砍舊 image → rebuild → 產生 shard → 跑 16 participant）
   ```bash
   docker compose down -v
   docker rmi -f $(docker images -q fl_spark* 2>/dev/null) || true
   docker compose build --no-cache data-init fl-server fl-participant-{1..16}
   docker compose up -d --force-recreate data-init       # 產生 shard
   docker logs -f data-init | cat                       # 等 "✓ Shards written"
   docker compose up -d fl-server fl-participant-{1..16}
   ```

### 二、性能觀察（傳統 FL vs Spark FL）
|  指標  | 傳統 FL  | Spark FL |
|--------|----------|----------|
| Round 1 時間 | 42.55 s | **35.11 s** |
| Round 20 累積 | 686.50 s | **609.98 s** |
| 最終測試準確率 | 98.22 % | **98.43 %** |

* 為什麼傳統 FL 越跑越慢？
   1. **序列化/網路**：每輪要 pickle 整個 `state_dict`，16 條 TCP 連線串行收發。
   2. **CUDA cache**：participant 每輪 `empty_cache()`，下一輪又得重新 warm-up。
   3. **Server thread 瓶頸**：`handle_client` 仍是 blocking 模式（見 `server.py`）。

> **20 s / 40 s 交錯現象？**

Round 時間在 `results/traditional/checkpoints/results.csv` 呈現出約 20 s 與 40 s 交錯的鋸齒型累積曲線，主要來自兩段「最慢者決定」的等待：

* **GPU 訓練階段 (~20 s)** — 16 個 participant 本地訓練 5 epoch，快的 17–19 s 就結束並把 update 傳回；慢的 (通常是第一批拿不到 GPU time-slice 或觸發 Lazy Tensor alloc) 會拖到 ~20 s ↑。
* **CPU 評估 + 圖表/I/O (~18 s)** — 服務端聚合後會：
  1. 在 CPU 上對 10 k MNIST 測試集推斷 (單執行緒約 14–16 s)。
  2. 用 Matplotlib 重繪雙軸性能圖並 `plt.savefig()` (2–3 s)。

若所有 participant 很快返回，Server 仍得跑完「評估 + 繪圖」，於是該輪總長逼近 40 s；反之只要有個 straggler 佔滿 20 s，Server 評估與繪圖與等待時間重疊，總長就壓到 ~20 s。兩種情況交錯，就得到 20 ↔ 40 s 的鋸齒型 timestamp。

改進方向：改用 asyncio/gRPC、持久化長連線、壓縮張量或改用 NCCL AllReduce。

## Fault-Tolerance Experiments Roadmap

為了系統化驗證容錯能力，後續所有實驗均放在頂層 `fault_tolerance/` 資料夾，每個子資料夾對應一個獨立場景，內含：
* `docker-compose.override.yml` — 只覆寫與故障相關的服務行為（建議用 `depends_on`, `command`, `profiles`）
* `run.sh` — 一鍵啟動 + 故障注入腳本
* `README.md` — 說明目標、步驟、評估指標、收集的 raw log 列表
* `results/` — 實驗輸出（⚠️ 勿 commit ※ Git ignore）

| Folder | Fault Model | 注入時機 | 注入機制 |
|---|---|---|---|
| `ft_node_loss_single/` | 單節點故障 | Round 5 | `docker stop fl-participant-9` 然後觀察自動重啟 |
| `ft_node_loss_multi/` | 同時 4 節點故障 | Round 10 | 停 `fl-participant-{5,6,11,12}` |
| `ft_shard_loss_10/` | 10% 資料缺失 | preprocess | 修改 `prepare_mnist.py` 隨機丟樣本 |
| `ft_shard_loss_30/` | 30% 資料缺失 | preprocess | 同上，但丟三成 |
| `ft_packet_loss_10/` | 10% 模型更新丟包 | transmit | 在 `participant.py` 內隨機 `continue` 傳送 |
| `ft_packet_loss_30/` | 30% 模型更新丟包 | transmit | 同上，30% |

### Spark FL 容錯優勢

| Fault Type | 傳統 FL 行為 | Spark FL 行為 | 為何 Spark 更佳 |
|------------|-------------|---------------|---------------|
| Node Loss  | Server 阻塞等待 / 需人工重啟 participant；整輪超時後才恢復 | Spark Task 失敗自動重排程到其他 executor，數秒內恢復 | Spark 的 Task speculation + heart-beat 檢測機制，無須人工干預 |
| Shard Loss | 單節點資料缺失→梯度偏差；需額外邏輯跳過空 shard | RDD 分區缺失自動重算（從 HDFS/cache），不影響聚合 | Spark 保留原始資料副本，多副本容錯 |
| Packet Loss | TCP socket 丟包→必須重傳整個 `state_dict` | 使用 Spark block transfer + chunk 重傳，僅補丟失 segment | Spark Netty 傳輸層具備 checksum 與 chunk-level retry |

**指標期待**
1. Node Loss: Spark FL 重啟時間 < 5 s；傳統 FL > 1 round (≈40 s)。
2. Shard Loss: Spark FL 減少 < 2% precision；傳統 FL 可能 > 5% 且易震盪。
3. Packet Loss: Spark FL throughput 下降 < 10%；傳統 FL 可能卡住或 timeout。

> 以上預期將在實驗結果中以 `results/*.csv` 及 Spark UI Event Logs 佐證。

> **執行範例（單節點故障）：**
>
> ```bash
> cd fault_tolerance/ft_node_loss_single
> ./run.sh          # 啟動 baseline + 定時 docker stop
> # 完成後結果位於 fault_tolerance/ft_node_loss_single/results/
> ```

所有新程式碼、Compose 檔只可放在 `fault_tolerance/`，嚴禁改動 `original/`。
