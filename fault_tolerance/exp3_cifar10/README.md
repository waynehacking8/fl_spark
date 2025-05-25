# 實驗三：CIFAR-10 數據集更換實驗

## 概述

本實驗是聯邦學習故障容錯研究的第三個實驗，**評估更大數據集（CIFAR-10）對Spark RDD容錯機制與傳統FL容錯機制性能差異的影響**。

相比MNIST實驗，CIFAR-10具有以下特點：
- **圖像尺寸**: 32×32×3 vs 28×28×1（數據量增加3.7倍）
- **複雜度**: 彩色自然圖像 vs 簡單手寫數字
- **計算需求**: 更高的內存和計算需求
- **容錯壓力**: 故障恢復開銷被放大

## 目錄結構

```
exp3_cifar10/
├── README.md                      # 本文檔
├── prepare_cifar10.py             # CIFAR-10數據準備腳本
├── models.py                      # CNN模型定義（適配CIFAR-10）
├── quick_test.py                  # 環境快速測試
├── traditional_code/              # 傳統FL實現
│   ├── server_cifar10.py         # FL服務器
│   └── participant_cifar10.py    # FL參與者
├── spark_code/                   # Spark FL實現
│   └── spark_fl_cifar10.py       # Spark聯邦學習
├── scripts/                      # 實驗腳本
│   └── run_experiment.sh         # 自動化實驗腳本
├── data/                         # 數據目錄
│   ├── raw/                      # 原始CIFAR-10數據
│   ├── cifar10_train_part1.pt    # 參與者1數據（25,000樣本）
│   ├── cifar10_train_part2.pt    # 參與者2數據（25,000樣本）
│   └── cifar10_test.pt           # 測試數據（10,000樣本）
└── results/                      # 實驗結果
    ├── traditional/              # 傳統FL結果
    ├── spark/                    # Spark FL結果
    └── cifar10_experiment_summary.txt  # 實驗摘要
```

## 環境要求

### 系統要求
- **內存**: 至少8GB RAM（推薦16GB）
- **存儲**: 約2GB空間（用於CIFAR-10數據和模型）
- **CPU**: 多核處理器（推薦4核+）

### Python依賴
```bash
pip install torch torchvision pyspark pandas matplotlib seaborn numpy
```

### 版本要求
- Python 3.7+
- PyTorch 1.8+
- PySpark 3.0+

## 快速開始

### 1. 環境測試
```bash
# 測試所有環境組件
python3 quick_test.py

# 應該看到所有測試通過：
# ✅ PyTorch環境: 通過
# ✅ PySpark環境: 通過  
# ✅ CIFAR-10下載: 通過
# ✅ 模型架構: 通過
# ✅ 數據準備: 通過
# ✅ 網絡端口: 通過
```

### 2. 數據準備
```bash
# 下載並分割CIFAR-10數據
python3 prepare_cifar10.py

# 生成的文件：
# data/cifar10_train_part1.pt  - 參與者1數據（25,000樣本）
# data/cifar10_train_part2.pt  - 參與者2數據（25,000樣本）
# data/cifar10_test.pt         - 測試數據（10,000樣本）
```

### 3. 運行完整實驗
```bash
# 自動化實驗（推薦）
chmod +x scripts/run_experiment.sh
./scripts/run_experiment.sh

# 分步執行
./scripts/run_experiment.sh deps        # 檢查依賴
./scripts/run_experiment.sh data        # 準備數據  
./scripts/run_experiment.sh traditional # 運行傳統FL
./scripts/run_experiment.sh spark       # 運行Spark FL
./scripts/run_experiment.sh analysis    # 生成分析
```

## 手動運行指南

### 傳統FL實驗

**1. 啟動服務器**
```bash
python3 traditional_code/server_cifar10.py \
    --participants 2 \
    --rounds 20 \
    --model simple \
    --timeout 30
```

**2. 啟動參與者（新終端）**
```bash
# 參與者1
python3 traditional_code/participant_cifar10.py 1 \
    --model simple \
    --batch_size 64 \
    --lr 0.001 \
    --epochs 5

# 參與者2（另一個終端）
python3 traditional_code/participant_cifar10.py 2 \
    --model simple \
    --batch_size 64 \
    --lr 0.001 \
    --epochs 5
```

### Spark FL實驗

```bash
python3 spark_code/spark_fl_cifar10.py \
    --participants 2 \
    --rounds 20 \
    --epochs 5 \
    --batch_size 64 \
    --lr 0.001 \
    --model simple
```

## 實驗設計

### 故障注入策略

**傳統FL**：第8輪參與者1模擬離線
- 服務器等待30秒超時檢測
- 使用參與者2的更新繼續訓練
- 測試checkpoint恢復機制

**Spark FL**：第8輪分區0模擬故障
- RDD血統追蹤自動檢測失敗partition
- 自動重新調度到可用executor
- 測試lineage自動重計算

### 實驗參數

| 參數 | 值 | 說明 |
|------|----|----- |
| 數據集 | CIFAR-10 | 50,000訓練+10,000測試 |
| 參與者數量 | 2 | 每個處理25,000樣本 |
| 聯邦輪數 | 20 | 完整訓練週期 |
| 本地輪數 | 5 | 每輪本地訓練次數 |
| 批次大小 | 64 | 適合32×32×3圖像 |
| 學習率 | 0.001 | Adam優化器 |
| 模型 | SimpleCNN | 3層卷積+2層全連接 |
| 故障輪次 | 8 | 中期故障注入 |

### 模型架構

**SimpleCNN (推薦)**:
- Conv2d(3→32) → Conv2d(32→64) → Conv2d(64→128)
- 3個MaxPool2d層，每層減半尺寸
- FC(2048→512) → FC(512→10)
- 參數量：~1.2M，計算快速

**StandardCNN (標準)**:
- 6層卷積 + BatchNorm + Dropout
- 3層全連接層
- 參數量：~2.5M，準確率更高

**ResNet (高級)**:
- 殘差連接 + 自適應池化
- 參數量：~400K，收斂穩定

## 評估指標

### 性能對比
- **訓練時間**: 總時間和每輪時間
- **最終準確率**: 20輪後的模型性能
- **收斂速度**: 達到目標準確率的輪數

### 故障容錯
- **故障檢測時間**: 發現參與者/分區故障的時間
- **恢復時間**: 從故障恢復到正常訓練的時間
- **性能影響**: 故障對準確率的影響程度
- **自動化程度**: 需要人工干預的程度

### 資源利用
- **內存使用**: 峰值內存占用
- **計算效率**: GPU/CPU利用率
- **網絡通信**: 數據傳輸量和頻率

## 預期結果

### CIFAR-10 vs MNIST 差異放大

**計算複雜度增加**:
- 圖像數據量: 3072 bytes vs 784 bytes（3.9倍）
- 模型參數量: ~1M vs ~100K（10倍）
- 內存需求: 顯著增加

**容錯機制壓力測試**:
- **RDD血統追蹤**: 在大數據量下的重計算效率
- **傳統checkpoint**: I/O開銷在大模型下的影響
- **網絡通信**: 模型參數傳輸的瓶頸

**預期發現**:
1. **Spark優勢放大**: RDD分區級容錯在大數據集下更高效
2. **傳統FL壓力暴露**: checkpoint I/O和網絡傳輸成為瓶頸
3. **故障恢復時間**: Spark自動恢復 vs 傳統手動重啟的差距拉大

## 結果文件說明

### CSV格式結果
```csv
Round,Timestamp,Accuracy,Loss,Participants,Failed_Participants
1,5.23,45.67,1.2345,2,0
8,45.67,78.90,0.8765,1,1  # 故障輪次
20,156.78,85.43,0.5432,2,0
```

### 關鍵指標分析
- **Round 8**: 故障檢測和處理輪次
- **Timestamp**: 累積運行時間（秒）
- **Accuracy**: 全局模型在測試集上的準確率（%）
- **Participants**: 成功參與訓練的參與者數量
- **Failed_Participants**: 故障參與者數量

## 故障排除

### 常見問題

**1. CUDA內存不足**
```bash
# 減小批次大小
python3 ... --batch_size 32

# 使用簡化模型
python3 ... --model simple
```

**2. PySpark啟動失敗**
```bash
# 檢查Java環境
java -version

# 設置較小內存
export PYSPARK_DRIVER_MEMORY=4g
export PYSPARK_EXECUTOR_MEMORY=4g
```

**3. 數據下載失敗**
```bash
# 手動下載CIFAR-10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# 解壓到 data/raw/
```

**4. 端口被占用**
```bash
# 查找占用進程
lsof -i :9999

# 殺死進程
kill -9 <PID>

# 或使用其他端口
python3 server_cifar10.py --port 9998
```

### 性能優化

**1. 加速訓練**
- 使用GPU: 確保 `torch.cuda.is_available()` 返回True
- 減少輪數: `--rounds 10` 用於快速測試
- 簡化模型: `--model simple`

**2. 內存優化**
- 減小批次: `--batch_size 32`
- 限制Spark內存: `spark.driver.memory=4g`
- 清理緩存: 定期重啟Jupyter/IDE

## 進階使用

### 自定義實驗參數

```bash
# 更多參與者（需準備更多數據分片）
./run_experiment.sh --participants 4

# 不同故障場景
# 修改 server_cifar10.py 中的 fault_round 和 failed_participants

# 不同模型對比
python3 spark_code/spark_fl_cifar10.py --model resnet
```

### 添加新模型

```python
# 在 models.py 中添加
class CustomCIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        # 自定義架構
        pass

# 在 get_model() 中註冊
elif model_type == 'custom':
    return CustomCIFAR10CNN(num_classes)
```

## 相關論文和參考

- **CIFAR-10數據集**: Krizhevsky, A. (2009). Learning multiple layers of features from tiny images.
- **聯邦學習**: McMahan, B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data.
- **Spark容錯**: Zaharia, M. et al. (2012). Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing.

## 許可證

本實驗代碼遵循MIT許可證。CIFAR-10數據集遵循其原始許可證。 