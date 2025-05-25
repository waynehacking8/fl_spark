#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import json
import csv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import traceback
import gc  # 添加垃圾回收模塊

# 檢測是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 清理CUDA緩存函數
def clean_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

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

# 數據轉換
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class SparkFL:
    def __init__(self, spark, num_workers=4, num_rounds=20, batch_size=32, learning_rate=0.01):
        self.spark = spark
        self.num_workers = num_workers  # 改為4個worker對應4個participant
        self.num_rounds = num_rounds  # 改為20輪
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # 設置設備
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # 使用第一個 GPU
            torch.cuda.empty_cache()
            print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("使用 CPU")
        
        self.model = CNNMnist().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_history = []
        self.data_dir = '/app/data'
        
        # 修改結果保存路徑，使用本地目錄
        self.results_dir = '/app/results/spark'
        os.makedirs(self.results_dir, exist_ok=True)
        self.accuracy_history_file = os.path.join(self.results_dir, 'spark_fl_accuracy.csv')
        self.results_file = os.path.join(self.results_dir, 'results.csv')
        self.performance_file = os.path.join(self.results_dir, 'performance.png')
        
        # 檢查點目錄
        self.checkpoint_dir = os.path.join(self.results_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.local_epochs = 5
        self.save_checkpoints = False
        self.enable_debug_logs = False
        
        # 加載測試集
        self.test_loader = self._get_test_loader()
        
        print("初始化 Spark 會話...")
        print(f"Spark 會話初始化完成。使用 Spark 版本: {self.spark.version}")
        
        # 初始化結果列表
        self.results = []
        
        # 故障模擬設置
        self.fault_round = 8  # 在第8輪注入故障
        self.failed_partitions = [0, 1]  # 模擬partition 0和1故障（對應participant 1&2）
        self.fault_timeout = 60  # 60秒超時
        
        # 初始化結果文件
        with open(self.results_file, 'w') as f:
            f.write('Round,Timestamp,Accuracy,Loss\n')
        
        if not os.path.exists(self.accuracy_history_file):
            with open(self.accuracy_history_file, 'w') as f:
                f.write('Round,Accuracy\n')
        
    def _log_memory_usage(self, stage):
        """記錄記憶體使用情況"""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**2
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"{stage} - 當前 GPU 記憶體使用: {current_memory:.2f}MB, 峰值: {peak_memory:.2f}MB")
            
    def _clean_memory(self):
        """清理記憶體"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._log_memory_usage("記憶體清理後")
        gc.collect()
        
    def _get_test_loader(self):
        """創建 MNIST 測試集的 DataLoader"""
        print(f"加載 MNIST 測試數據...")
        test_dataset = datasets.MNIST(self.data_dir, train=False, download=True, transform=mnist_transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        print(f"測試數據加載完成。樣本數: {len(test_dataset)}")
        return test_loader
        
    def _save_accuracy_history(self):
        """將準確率歷史保存到 CSV 文件"""
        if not self.accuracy_history:
            print("沒有準確率歷史記錄需要保存")
            return
            
        try:
            print(f"保存準確率歷史記錄到 {self.accuracy_history_file}")
            with open(self.accuracy_history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Round', 'Accuracy'])  # 寫入標題
                writer.writerows(self.accuracy_history)
            print("準確率歷史記錄已保存")
        except Exception as e:
            print(f"保存準確率歷史記錄時出錯: {e}")
    
    def _save_results(self, round_num, accuracy, loss, timestamp):
        """保存訓練結果到 CSV 文件和內存"""
        # 保存到內存
        self.results.append({
            'round': round_num,
            'timestamp': timestamp,
            'accuracy': accuracy,
            'loss': loss
        })
        
        # 追加到文件
        with open(self.results_file, 'a') as f:
            f.write(f"{round_num},{timestamp},{accuracy},{loss}\n")
        print(f"第 {round_num} 輪結果已保存：準確率 {accuracy:.2f}%, 損失 {loss:.4f}, 用時 {timestamp:.2f} 秒")
    
    def _plot_performance(self):
        """繪製性能圖表"""
        import matplotlib.pyplot as plt
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # 從內存中讀取數據
        rounds = [r['round'] for r in self.results]
        accuracies = [r['accuracy'] for r in self.results]
        losses = [r['loss'] for r in self.results]
        
        # 繪製準確率曲線
        ax1.plot(rounds, accuracies, 'r-o', label='Accuracy')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Training Accuracy over Rounds')
        ax1.grid(True)
        ax1.legend()
        
        # 設置準確率範圍從 80% 到 100%
        ax1.set_ylim(80, 100)
        
        # 繪製損失曲線
        ax2.plot(rounds, losses, 'b-x', label='Loss')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss over Rounds')
        ax2.grid(True)
        ax2.legend()
        
        # 調整子圖之間的間距
        plt.tight_layout()
        
        # 保存圖表
        plt.savefig(self.performance_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"性能圖表已更新：{self.performance_file}")
    
    def train(self):
        """使用 Spark 執行聯邦學習訓練"""
        # 直接使用預處理的數據，避免重複下載
        print("使用預處理的數據分片，避免重複下載 MNIST...")
        total_samples = 60000  # MNIST 數據集總樣本數
        print(f"訓練數據總樣本數: {total_samples}")
        
        # 創建固定的數據分片
        samples_per_partition = total_samples // self.num_workers
        print(f"創建 {self.num_workers} 個分區，每個約 {samples_per_partition} 個樣本")
        
        # 創建固定的分片索引
        partition_indices = []
        for i in range(self.num_workers):
            start_idx = i * samples_per_partition
            end_idx = start_idx + samples_per_partition if i < self.num_workers - 1 else total_samples
            partition_indices.append(list(range(start_idx, end_idx)))
            print(f"分區 {i} 的索引範圍: {start_idx} 到 {end_idx-1}, 樣本數: {end_idx-start_idx}")
        
        # 創建分區 RDD
        index_rdd = self.spark.sparkContext.parallelize(partition_indices, self.num_workers)
        actual_partitions = index_rdd.getNumPartitions()
        print(f"實際創建的分區數: {actual_partitions}")
        
        # 開始訓練循環
        print(f"開始訓練，共 {self.num_rounds} 個 round")
        start_time = time.time()
        
        for round in range(self.num_rounds):
            round_num = round + 1
            print(f"\n===== 開始 Round {round_num}/{self.num_rounds} =====")
            round_start_time = time.time()
            
            # 清理記憶體
            self._clean_memory()
            
            # 將當前模型參數序列化為字典以便廣播
            model_params = {}
            for key, value in self.model.state_dict().items():
                model_params[key] = value.cpu().numpy()
            
            # 廣播模型參數
            broadcast_model = self.spark.sparkContext.broadcast(model_params)
            
            # 在每個分區上執行本地訓練
            print(f"Round {round_num}: 開始分布式訓練...")
            
            def local_train(indices_iterator):
                """在 Worker 上執行本地訓練"""
                import os
                import numpy as np
                import torch
                import torch.nn as nn
                import torch.optim as optim
                from torchvision import datasets, transforms
                from torch.utils.data import Subset, DataLoader
                import gc
                import time
                
                # 設置設備
                worker_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if torch.cuda.is_available():
                    torch.cuda.set_device(0)
                    torch.cuda.empty_cache()
                
                # 收集分區內的所有索引
                indices = []
                for idx_list in indices_iterator:
                    indices.extend(idx_list)
                
                if not indices:
                    return [(0, None)]
                
                # 獲取分區ID（通過檢查索引範圍判斷）
                min_idx = min(indices)
                partition_id = min_idx // 15000  # 每個分區15000個樣本
                
                worker_id = os.getpid()
                print(f"Worker PID: {worker_id}, 分區 {partition_id} 開始處理 {len(indices)} 個樣本")
                
                # 故障注入：如果是第8輪且為故障分區，模擬節點故障
                current_round = round_num  # 從外層作用域獲取
                if current_round == 8 and partition_id in [0, 1]:  # 模擬participant 1&2故障
                    print(f"Worker PID: {worker_id}, 分區 {partition_id} 模擬故障，60秒後超時")
                    time.sleep(60)  # 模擬節點無響應，觸發超時機制
                    raise Exception(f"Worker節點故障：分區 {partition_id} 在第8輪離線")
                
                print(f"Worker PID: {worker_id}, 分區 {partition_id} 正常處理 {len(indices)} 個樣本")
                
                try:
                    # 定義 CNN 模型
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
                    
                    # 初始化模型
                    model = CNNMnist().to(worker_device)
                    
                    # 從廣播變量加載模型參數
                    np_params = broadcast_model.value
                    state_dict = {}
                    for key, value in np_params.items():
                        state_dict[key] = torch.tensor(value).to(worker_device)
                    model.load_state_dict(state_dict)
                    
                    # 加載數據
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
                    
                    # 根據分區ID加載對應的數據文件
                    data_files = [
                        '/app/data/mnist_train_part1.pt',  # 分區0：participant 1
                        '/app/data/mnist_train_part2.pt',  # 分區1：participant 2  
                        '/app/data/mnist_train_part3.pt',  # 分區2：participant 3
                        '/app/data/mnist_train_part4.pt'   # 分區3：participant 4
                    ]
                    
                    train_data = torch.load(data_files[partition_id])
                    data = train_data['data'].float() / 255.0
                    targets = train_data['targets']
                    
                    # 創建本地數據集 - 修正索引邏輯
                    # 將全局索引轉換為相對索引
                    local_indices = []
                    partition_start = partition_id * 15000
                    for idx in indices:
                        if partition_start <= idx < partition_start + 15000:
                            local_indices.append(idx - partition_start)
                    
                    if not local_indices:
                        print(f"Worker PID: {worker_id}, 分區 {partition_id} 沒有有效數據")
                        return [(0, None)]
                    
                    local_data = data[local_indices]
                    local_targets = targets[local_indices]
                    
                    # 調整數據形狀以匹配模型輸入
                    local_data = local_data.unsqueeze(1)  # 添加通道維度
                    
                    # 創建 DataLoader
                    local_dataset = torch.utils.data.TensorDataset(local_data, local_targets)
                    local_loader = DataLoader(
                        local_dataset, 
                        batch_size=64,  # 增加批次大小
                        shuffle=True,
                        pin_memory=False  # 在容器環境中關閉pin_memory
                    )
                    
                    print(f"Worker PID: {worker_id} 數據加載成功，全局索引 {len(indices)} 個，本地樣本 {len(local_dataset)} 個")
                    
                    # 設置訓練
                    model.train()
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
                    
                    # 執行多輪本地訓練
                    for local_epoch in range(5):
                        print(f"Worker PID: {worker_id} 開始本地訓練輪 {local_epoch+1}/5")
                        for batch_idx, (data, target) in enumerate(local_loader):
                            # 將數據移到正確的設備
                            data, target = data.to(worker_device), target.to(worker_device)
                            
                            # 前向傳播與反向傳播
                            optimizer.zero_grad()
                            output = model(data)
                            loss = criterion(output, target)
                            loss.backward()
                            optimizer.step()
                            
                            # 每 50 個批次清理一次記憶體
                            if batch_idx % 50 == 0:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                gc.collect()
                                print(f"Worker PID: {worker_id} 批次 {batch_idx}/{len(local_loader)}")
                        
                        # 清理記憶體
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                    
                    # 返回更新後的模型參數
                    updated_params = {}
                    for key, value in model.state_dict().items():
                        updated_params[key] = value.cpu().numpy()
                    
                    print(f"Worker PID: {worker_id} 訓練完成，返回更新")
                    return [(len(local_indices), updated_params)]
                    
                except Exception as e:
                    print(f"Worker PID: {worker_id} 訓練錯誤: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return [(0, None)]
            
            # 在每個分區上執行訓練並收集結果
            try:
                updates_rdd = index_rdd.mapPartitions(local_train)
                
                # 設置任務超時（特別是第8輪）
                if round_num == self.fault_round:
                    print(f"Round {round_num}: 檢測到故障輪次，設置 {self.fault_timeout} 秒超時")
                    # 設置 Spark 任務超時
                    spark.conf.set("spark.task.maxFailures", "1")
                    spark.conf.set("spark.stage.maxConsecutiveAttempts", "1")
                
                start_collect_time = time.time()
                collected_updates = updates_rdd.collect()
                collect_time = time.time() - start_collect_time
                
                # 過濾有效的更新
                valid_updates = [(samples, params) for samples, params in collected_updates if params is not None]
                total_samples = sum(samples for samples, _ in valid_updates)
                
                if round_num == self.fault_round:
                    print(f"Round {round_num}: 故障檢測完成，收集時間 {collect_time:.2f} 秒")
                    if len(valid_updates) < self.num_workers:
                        print(f"Round {round_num}: 檢測到 Worker 節點故障，{self.num_workers - len(valid_updates)} 個分區失效")
                        print(f"Round {round_num}: RDD 血統追蹤自動重計算失敗的分區")
                
                print(f"Round {round_num}: 收到 {len(valid_updates)}/{len(collected_updates)} 個有效更新，"
                      f"總樣本數 {total_samples}")
                
                if valid_updates and total_samples > 0:
                    print(f"Round {round_num}: 執行聯邦平均...")
                    
                    # 初始化聚合字典
                    aggregated_params = {}
                    
                    # 聯邦平均
                    for key in valid_updates[0][1].keys():
                        # 構建加權平均
                        weighted_sum = np.zeros_like(valid_updates[0][1][key])
                        for samples, params in valid_updates:
                            weight = samples / total_samples
                            weighted_sum += params[key] * weight
                        
                        # 轉換為 Torch 張量
                        aggregated_params[key] = torch.tensor(weighted_sum)
                    
                    # 更新全局模型
                    self.model.load_state_dict(aggregated_params)
                    print(f"Round {round_num}: 全局模型已更新")
                else:
                    print(f"Round {round_num}: 沒有有效的更新，全局模型未更新")
                
                # 只在訓練結束時保存最終模型
                if round_num == self.num_rounds and self.save_checkpoints:
                    checkpoint_path = os.path.join(self.checkpoint_dir, f'final_model.pt')
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print(f"訓練完成：最終模型已保存到 {checkpoint_path}")
                
                # 評估當前模型
                accuracy, loss = self._test_model()
                self.accuracy_history.append([round_num, accuracy])
                
                # 保存當前輪次的結果
                timestamp = time.time() - start_time
                self._save_results(round_num, accuracy, loss, timestamp)
                self._save_accuracy_history()
                
                # 每輪都更新性能圖表
                self._plot_performance()
                
                # 記錄 round 用時
                round_time = time.time() - round_start_time
                print(f"===== Round {round_num} 完成，用時 {round_time:.2f} 秒，準確率 {accuracy:.2f}% =====")
            
            except Exception as e:
                print(f"Round {round_num} 執行期間發生錯誤: {str(e)}")
                traceback.print_exc()
                # 即使發生錯誤，也保存當前輪次的結果
                accuracy, loss = self._test_model()
                self.accuracy_history.append([round_num, accuracy])
                timestamp = time.time() - start_time
                self._save_results(round_num, accuracy, loss, timestamp)
                self._save_accuracy_history()
            
            # 釋放記憶體
            del broadcast_model
            self._clean_memory()
            
            # 等待 1 秒再進入下一輪
            time.sleep(1)
        
        if self.accuracy_history:
            print(f"訓練完成，最終準確率: {self.accuracy_history[-1][1]:.2f}%")
            print(f"總用時: {time.time() - start_time:.2f} 秒")
    
    def _test_model(self):
        """評估全局模型在測試集上的準確率"""
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            for data, target in self.test_loader:
                # 將數據移動到正確的設備
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                
                # 計算損失
                loss = criterion(output, target).item()
                total_loss += loss
                
                # 計算正確預測
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        # 計算準確率和平均損失
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / total
        print(f"測試結果: {correct}/{total} 正確 ({accuracy:.2f}%)")
        
        # 重設為訓練模式
        self.model.train()
        return accuracy, avg_loss
        
    def stop(self):
        """停止 Spark 會話並保存結果"""
        self._save_accuracy_history()
        print("停止 Spark 會話...")
        self.spark.stop()
        print("Spark 會話已停止")
        
    def _load_accuracy_history(self):
        """從 CSV 文件加載準確率歷史記錄（如果存在）"""
        if os.path.exists(self.accuracy_history_file):
            try:
                print(f"加載現有準確率歷史記錄從 {self.accuracy_history_file}")
                with open(self.accuracy_history_file, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader)  # 跳過標題行
                    if header == ['Round', 'Accuracy']:
                        self.accuracy_history = [[int(row[0]), float(row[1])] for row in reader]
                        print(f"已加載 {len(self.accuracy_history)} 條記錄")
                    else:
                        print("文件標題不匹配，創建新的歷史記錄")
                        self.accuracy_history = []
            except Exception as e:
                print(f"加載準確率歷史記錄出錯: {e}, 創建新的歷史記錄")
                self.accuracy_history = []
        else:
            print("未找到現有準確率歷史文件")
            self.accuracy_history = []

if __name__ == "__main__":
    # 創建 SparkFL 實例並運行訓練
    spark = SparkSession.builder \
        .appName("FederatedLearningSpark") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.cores", "2") \
        .config("spark.python.worker.memory", "8g") \
        .config("spark.default.parallelism", "4") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.3") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "1024m") \
        .config("spark.kryoserializer.buffer", "64m") \
        .config("spark.shuffle.compress", "true") \
        .config("spark.broadcast.compress", "true") \
        .config("spark.rdd.compress", "true") \
        .config("spark.io.compression.codec", "snappy") \
        .config("spark.local.dir", "/tmp") \
        .config("spark.worker.cleanup.enabled", "true") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:MaxGCPauseMillis=200") \
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:MaxGCPauseMillis=200") \
        .config("spark.task.maxFailures", "3") \
        .config("spark.stage.maxConsecutiveAttempts", "4") \
        .config("spark.executor.heartbeatInterval", "10s") \
        .config("spark.network.timeout", "600s") \
        .getOrCreate()
    
    # 設置日誌級別
    spark.sparkContext.setLogLevel("ERROR")
    
    spark_fl = SparkFL(spark, num_workers=4, num_rounds=20)
    try:
        spark_fl.train()
    except Exception as e:
        print(f"訓練過程中發生錯誤: {e}")
        traceback.print_exc()
    finally:
        spark_fl.stop() 