# ft_node_loss_single

單節點故障測試：Round 5 時停止 `fl-participant-9`，觀察系統是否繼續收斂。

## 步驟
```bash
cd fault_tolerance/ft_node_loss_single
./run.sh               # 預設 120s 後注入故障
# 若要自訂延遲或目標容器
DELAY=150 TARGET=fl-participant-4 ./run.sh
```

## 指標
* 訓練是否能跑滿 20 rounds
* Round 5 之後 loss / accuracy 曲線是否平滑
* `fl-server` 日誌：能否偵測缺席 participant 而繼續聚合

## 輸出
* `results/` 內自動複製 server 產生的 `results.csv`, `performance.png`
