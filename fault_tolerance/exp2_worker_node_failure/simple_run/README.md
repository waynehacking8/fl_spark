# EXP2 Worker節點故障容錯實驗

## 📁 文件結構

```
simple_run/
├── 🚀 run_simultaneous_experiment.sh    # 主實驗啟動腳本
├── 🖥️  server_fixed.py                  # 聯邦學習服務器
├── 👥 participant_fixed.py              # 參與者客戶端
├── 📊 prepare_data.py                   # 數據準備腳本
├── 📈 plot_results.py                   # 性能圖表生成
├── 📉 plot_training_progress.py         # 訓練進度圖表
├── 📝 README.md                        # 使用說明
├── data/                               # 數據目錄
└── results/traditional/checkpoints/    # 結果輸出
```

## 🚀 快速啟動

```bash
# 1. 準備數據
python prepare_data.py

# 2. 運行實驗
./run_simultaneous_experiment.sh

# 3. 生成圖表
python plot_results.py
python plot_training_progress.py
```

## 📊 實驗特色

- ✅ **第8輪故障容錯**：自動檢測參與者1&2離線
- ✅ **60秒超時機制**：快速故障檢測
- ✅ **自動編號修正**：保證輪次連續性(1-19)
- ✅ **同時啟動優化**：減少第一輪延遲至8秒
- ✅ **Checkpoint恢復**：故障後快速恢復訓練

## 📈 預期結果

- 總輪次：19輪（含故障恢復）
- 最終準確率：99%+
- 故障檢測：第8輪自動跳過
- 圖表輸出：performance.png + traditional_fl_training_progress.png

詳細說明請參考：`../EXPERIMENT_REPORT.md` 