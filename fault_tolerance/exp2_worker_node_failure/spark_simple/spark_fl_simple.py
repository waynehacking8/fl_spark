#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spark FL 簡化版本 - Worker Node Fault Tolerance 實驗
不依賴 Docker Compose，直接運行本地 Spark
"""
import os
import sys
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import traceback
import gc

# 檢查和安裝 PySpark
try:
    from pyspark.sql import SparkSession
    from pyspark import SparkContext, SparkConf
except ImportError:
    print("安裝 PySpark...")
    os.system("pip install pyspark==3.4.1")
    from pyspark.sql import SparkSession
    from pyspark import SparkContext, SparkConf

# CNN 模型定義
class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleSparkFL:
    def __init__(self, num_participants=4, num_rounds=20, local_epochs=5):
        self.num_participants = num_participants
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        
        # 設備設置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用設備: {self.device}")
        
        # 初始化模型
        self.global_model = CNNMnist().to(self.device)
        
        # 創建結果目錄
        os.makedirs('results', exist_ok=True)
        self.results_file = 'results/spark_fl_results.csv'
        self.accuracy_file = 'results/spark_fl_accuracy.csv'
        
        # 初始化結果文件
        with open(self.results_file, 'w') as f:
            f.write('Round,Timestamp,Accuracy,Loss,Participants,Failed_Participants\n')
        
        # 故障注入設置
        self.fault_round = 8
        self.failed_participants = [0, 1]  # 參與者1和2在第8輪故障
        
        # 加載測試數據
        self.test_data = self._load_test_data()
        
        print(f"Spark FL 初始化完成")
        print(f"參與者數量: {self.num_participants}")
        print(f"訓練輪數: {self.num_rounds}")
        print(f"本地訓練輪數: {self.local_epochs}")
        print(f"故障輪次: {self.fault_round} (參與者 {[p+1 for p in self.failed_participants]} 故障)")
    
    def _load_test_data(self):
        """加載測試數據"""
        test_file = 'data/test_data.pt'
        if os.path.exists(test_file):
            test_data = torch.load(test_file)
            print(f"測試數據加載完成: {len(test_data['data'])} 樣本")
            return test_data
        else:
            print("錯誤: 測試數據文件不存在，請先運行 prepare_data.py")
            sys.exit(1)
    
    def _create_spark_session(self):
        """創建本地 Spark 會話"""
        print("創建本地 Spark 會話...")
        
        spark = SparkSession.builder \
            .appName("SimpleSparkFL_WorkerFaultTolerance") \
            .master("local[4]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.default.parallelism", "4") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("ERROR")
        print(f"Spark 會話創建成功，版本: {spark.version}")
        return spark
    
    def _test_model(self):
        """測試全局模型性能"""
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        # 創建測試 DataLoader
        test_dataset = TensorDataset(self.test_data['data'], self.test_data['targets'])
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                _, predicted = torch.max(output.data, 1)
                
                loss = criterion(output, target).item()
                total_loss += loss
                
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / total
        
        self.global_model.train()
        return accuracy, avg_loss
    
    def _local_training(self, participant_id_data):
        """在 Spark worker 上執行本地訓練"""
        participant_id, global_params = participant_id_data
        
        try:
            # 設置設備
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            print(f"參與者 {participant_id+1} 開始本地訓練...")
            
            # 故障注入
            if hasattr(self, '_current_round') and self._current_round == self.fault_round and participant_id in self.failed_participants:
                print(f"參與者 {participant_id+1} 在第{self.fault_round}輪模擬故障")
                time.sleep(30)  # 模擬30秒延遲
                raise Exception(f"參與者 {participant_id+1} 節點故障")
            
            # 創建本地模型
            model = CNNMnist().to(device)
            
            # 加載全局模型參數
            state_dict = {}
            for key, value in global_params.items():
                state_dict[key] = torch.tensor(value).to(device)
            model.load_state_dict(state_dict)
            
            # 加載參與者數據
            data_file = f'data/participant_{participant_id+1}_data.pt'
            participant_data = torch.load(data_file)
            
            # 創建 DataLoader
            dataset = TensorDataset(participant_data['data'], participant_data['targets'])
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            # 設置訓練
            model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
            
            # 本地訓練
            for epoch in range(self.local_epochs):
                for batch_idx, (data, target) in enumerate(dataloader):
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    if batch_idx % 50 == 0:
                        print(f"參與者 {participant_id+1}, Epoch {epoch+1}/{self.local_epochs}, Batch {batch_idx}")
            
            # 返回更新後的模型參數
            updated_params = {}
            for key, value in model.state_dict().items():
                updated_params[key] = value.cpu().numpy()
            
            print(f"參與者 {participant_id+1} 本地訓練完成")
            return (participant_id, len(dataset), updated_params)
            
        except Exception as e:
            print(f"參與者 {participant_id+1} 訓練失敗: {str(e)}")
            return (participant_id, 0, None)
    
    def run_experiment(self):
        """運行完整的實驗"""
        print("開始 Spark FL Worker Node Fault Tolerance 實驗")
        
        # 創建 Spark 會話
        spark = self._create_spark_session()
        
        start_time = time.time()
        
        try:
            for round_num in range(1, self.num_rounds + 1):
                print(f"\n{'='*50}")
                print(f"Round {round_num}/{self.num_rounds}")
                print(f"{'='*50}")
                
                round_start = time.time()
                self._current_round = round_num
                
                # 準備全局模型參數
                global_params = {}
                for key, value in self.global_model.state_dict().items():
                    global_params[key] = value.cpu().numpy()
                
                # 創建參與者數據 RDD
                participant_data = [(i, global_params) for i in range(self.num_participants)]
                participant_rdd = spark.sparkContext.parallelize(participant_data, self.num_participants)
                
                # 執行分布式訓練
                if round_num == self.fault_round:
                    print(f"Round {round_num}: 注入故障 - 參與者 {[p+1 for p in self.failed_participants]} 將故障")
                
                training_results = participant_rdd.map(self._local_training).collect()
                
                # 處理訓練結果
                valid_results = [(pid, samples, params) for pid, samples, params in training_results if params is not None]
                failed_participants = [pid+1 for pid, samples, params in training_results if params is None]
                
                print(f"Round {round_num}: {len(valid_results)}/{self.num_participants} 參與者成功完成訓練")
                if failed_participants:
                    print(f"Round {round_num}: 失敗的參與者: {failed_participants}")
                
                # 聯邦平均
                if valid_results:
                    total_samples = sum(samples for _, samples, _ in valid_results)
                    aggregated_params = {}
                    
                    # 執行加權平均
                    for key in valid_results[0][2].keys():
                        weighted_sum = np.zeros_like(valid_results[0][2][key])
                        for _, samples, params in valid_results:
                            weight = samples / total_samples
                            weighted_sum += params[key] * weight
                        aggregated_params[key] = torch.tensor(weighted_sum)
                    
                    # 更新全局模型
                    self.global_model.load_state_dict(aggregated_params)
                    print(f"Round {round_num}: 全局模型已更新 (基於 {len(valid_results)} 個參與者)")
                else:
                    print(f"Round {round_num}: 沒有有效更新，跳過模型聚合")
                
                # 測試模型性能
                accuracy, loss = self._test_model()
                
                # 記錄結果
                round_time = time.time() - round_start
                timestamp = time.time() - start_time
                
                with open(self.results_file, 'a') as f:
                    f.write(f"{round_num},{timestamp:.2f},{accuracy:.2f},{loss:.4f},{len(valid_results)},{len(failed_participants)}\n")
                
                print(f"Round {round_num} 完成:")
                print(f"  準確率: {accuracy:.2f}%")
                print(f"  損失: {loss:.4f}")
                print(f"  用時: {round_time:.2f} 秒")
                print(f"  成功參與者: {len(valid_results)}/{self.num_participants}")
                
                # 清理記憶體
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                time.sleep(1)  # 短暫休息
        
        finally:
            spark.stop()
            total_time = time.time() - start_time
            print(f"\n實驗完成，總用時: {total_time:.2f} 秒")
            print(f"結果已保存到: {self.results_file}")

def main():
    # 檢查數據是否存在
    if not os.path.exists('data/participant_1_data.pt'):
        print("數據文件不存在，正在準備數據...")
        exec(open('prepare_data.py').read())
    
    # 運行實驗
    spark_fl = SimpleSparkFL()
    spark_fl.run_experiment()

if __name__ == "__main__":
    main() 