#!/bin/bash

# 創建數據目錄
mkdir -p data/mnist

# 安裝依賴
pip install -r requirements.txt

# 啟動服務器
python traditional_code/server.py &

# 等待服務器啟動
sleep 2

# 啟動參與者
for i in {0..9}; do
    python traditional_code/participant.py $i &
done

# 等待所有進程完成
wait 