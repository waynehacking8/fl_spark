# EXP2: Worker節點故障容錯聯邦學習實驗

## 📋 實驗簡介

本實驗驗證傳統聯邦學習在Worker節點故障情況下的容錯能力。模擬第8輪訓練時參與者1&2同時離線的故障場景，測試故障檢測、恢復機制和訓練連續性。

## 📁 項目結構

```
exp2_worker_node_failure/
├── 📋 EXPERIMENT_REPORT.md     # 詳細實驗報告
├── 📝 README.md                # 項目說明（本文件）
├── 📦 requirements.txt         # Python依賴包
└── 🚀 simple_run/              # 實驗執行目錄
    ├── run_simultaneous_experiment.sh  # 主啟動腳本
    ├── server_fixed.py          # 聯邦學習服務器
    ├── participant_fixed.py     # 參與者客戶端
    ├── prepare_data.py          # 數據準備腳本
    ├── plot_results.py          # 性能圖表生成
    ├── plot_training_progress.py # 訓練進度圖表
    ├── README.md                # 快速使用指南
    ├── data/                    # 數據目錄
    └── results/                 # 結果輸出目錄
```

## 🚀 快速開始

```bash
# 1. 進入實驗目錄
cd simple_run/

# 2. 安裝依賴
pip install -r ../requirements.txt

# 3. 準備數據
python prepare_data.py

# 4. 運行實驗
./run_simultaneous_experiment.sh

# 5. 生成圖表
python plot_results.py
python plot_training_progress.py
```

## 🎯 實驗亮點

- ✅ **第8輪故障容錯**：自動檢測參與者離線
- ✅ **60秒超時機制**：快速故障檢測  
- ✅ **自動編號修正**：保證輪次連續性(1-19)
- ✅ **Checkpoint恢復**：故障後快速恢復訓練
- ✅ **同時啟動優化**：減少第一輪延遲

## 📊 預期結果

- **總輪次**：19輪（含故障恢復）
- **最終準確率**：99%+
- **故障檢測時間**：60秒（超時機制）
- **恢復時間**：<5秒（checkpoint載入）

## 📖 詳細說明

請參閱 [`EXPERIMENT_REPORT.md`](./EXPERIMENT_REPORT.md) 了解：
- 詳細技術原理
- 容錯機制設計
- 實驗結果分析
- 應用場景說明

---

**技術棧**：Python, PyTorch, Socket編程, MNIST數據集  
**容錯機制**：超時檢測 + Checkpoint恢復 + 自動編號修正 