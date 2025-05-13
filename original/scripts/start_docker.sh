#!/bin/bash

# 停止並刪除現有容器
docker-compose down

# 構建並啟動容器
docker-compose up --build -d

# 等待 Spark 集群啟動
echo "等待 Spark 集群啟動..."
sleep 10

# 啟動傳統聯邦學習系統
echo "啟動傳統聯邦學習系統..."
docker-compose exec fl-server python /app/traditional_code/server.py &
sleep 2
docker-compose exec fl-participant-1 python /app/traditional_code/participant.py 0 &
docker-compose exec fl-participant-2 python /app/traditional_code/participant.py 1 &

# 啟動 Spark 聯邦學習系統
echo "啟動 Spark 聯邦學習系統..."
docker-compose exec spark-master python /app/spark_code/main.py

# 等待所有進程完成
wait 