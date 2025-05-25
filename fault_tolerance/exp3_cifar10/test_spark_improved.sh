#!/bin/bash

echo "=== 測試改進版Spark FL (10輪) ==="

cd spark_code

# 運行改進版Spark FL
echo "啟動改進版Spark FL..."
python spark_fl_cifar10.py --rounds 10 --model resnet --mode normal

echo ""
echo "=== 結果對比 ==="
echo ""

echo "原版Spark FL結果："
tail -5 ../results/spark/normal/cifar10_spark_normal_results.csv

echo ""
echo "改進版Spark FL結果："
tail -5 ../results/spark_improved/normal/cifar10_spark_improved_normal_results.csv

echo ""
echo "測試完成！" 