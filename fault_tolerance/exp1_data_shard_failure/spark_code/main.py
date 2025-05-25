import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
import pandas as pd
import gc
import csv
import traceback

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/results/spark_fl.log')
    ]
)
logger = logging.getLogger('SparkFL')

# 故障模擬參數
FAILURE_ROUND = 5
SIMULATE_FAILURE = True

# 檢測是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

def clean_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# 數據轉換
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

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

class SparkFL:
    def __init__(self, spark, num_workers=2, num_rounds=20, batch_size=32, learning_rate=0.01):
        self.spark = spark
        self.num_workers = num_workers
        self.num_rounds = num_rounds
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
        
        # 修改結果保存路徑，使用相對於當前目錄的路徑
        self.results_dir = '../results/spark'
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
        
    def _save_accuracy_history(self, round_num, accuracy):
        """保存當前輪次的準確率到 CSV 文件"""
        with open(self.accuracy_history_file, 'a') as f:
                f.write(f"{round_num},{accuracy}\n")
        print(f"第 {round_num} 輪準確率已保存到 {self.accuracy_history_file}")
    
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
        # 加載訓練數據以獲取總數
        print("加載訓練數據以獲取總數...")
        train_dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=mnist_transform)
        total_samples = len(train_dataset)
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
                
                worker_id = os.getpid()
                
                # 故障模擬：簡單方式
                from pyspark import TaskContext
                context = TaskContext.get()
                partition_id = context.partitionId() if context else 0
                
                if round_num == FAILURE_ROUND and SIMULATE_FAILURE and partition_id == 0:
                    print(f"模擬分區 {partition_id} 在第 {round_num} 輪發生數據訪問故障")
                    # 返回失敗結果而不是拋出異常
                    return [(0, None)]
                
                print(f"Worker PID: {worker_id} 開始處理 {len(indices)} 個樣本")
                
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
                    
                    # 使用完整的預處理數據文件
                    train_data = torch.load('/app/data/mnist_train_complete.pt')
                    data = train_data['data'].float() / 255.0
                    targets = train_data['targets']
                    
                    # 創建本地數據集（根據分配的索引）
                    local_data = data[indices]
                    local_targets = targets[indices]
                    
                    # 調整數據形狀以匹配模型輸入
                    local_data = local_data.unsqueeze(1)  # 添加通道維度
                    
                    # 創建 DataLoader
                    local_dataset = torch.utils.data.TensorDataset(local_data, local_targets)
                    local_loader = DataLoader(
                        local_dataset, 
                        batch_size=64,  # 增加批次大小
                        shuffle=True,
                        pin_memory=True,
                        num_workers=2  # 使用多進程加載數據
                    )
                    
                    print(f"Worker PID: {worker_id} 數據加載成功，共 {len(local_dataset)} 個樣本")
                    
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
                    return [(len(indices), updated_params)]
                    
                except Exception as e:
                    print(f"Worker PID: {worker_id} 訓練錯誤: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return [(0, None)]
            
            # 在每個分區上執行訓練並收集結果
            collected_updates = []
            try:
                updates_rdd = index_rdd.mapPartitions(local_train)
                collected_updates = updates_rdd.collect()
                
            except Exception as e:
                print(f"Round {round_num}: 部分分區失敗: {str(e)}")
                print(f"Round {round_num}: 使用部分結果繼續訓練...")
                
                # 嘗試手動收集可用的結果
                try:
                    # 重新嘗試收集，但允許部分失敗
                    updates_rdd = index_rdd.mapPartitions(local_train)
                    collected_updates = updates_rdd.collect()
                except:
                    # 如果還是失敗，就用空結果繼續
                    print(f"Round {round_num}: 無法收集任何結果，跳過此輪")
                    collected_updates = [(0, None)] * self.num_workers
                
                # 過濾有效的更新
                valid_updates = [(samples, params) for samples, params in collected_updates if params is not None]
                total_samples = sum(samples for samples, _ in valid_updates)
                
                print(f"Round {round_num}: 收到 {len(valid_updates)}/{len(collected_updates)} 個有效更新，"
                      f"總樣本數 {total_samples}")
                
            # 要求所有workers都成功才進行聚合
            if len(valid_updates) == self.num_workers and total_samples > 0:
                print(f"Round {round_num}: 所有workers都成功，執行聯邦平均...")
                    
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
                print(f"Round {round_num}: 只有 {len(valid_updates)}/{self.num_workers} 個workers成功，跳過此輪聚合")
                print(f"Round {round_num}: 全局模型保持不變")
                
                # 只在訓練結束時保存最終模型
                if round_num == self.num_rounds and self.save_checkpoints:
                    checkpoint_path = os.path.join(self.checkpoint_dir, f'final_model.pt')
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print(f"訓練完成：最終模型已保存到 {checkpoint_path}")
                
                # 評估當前模型
                accuracy, loss = self._test_model()
                self.accuracy_history.append([round_num, accuracy])
            self._save_accuracy_history(round_num, accuracy)
                
                # 保存當前輪次的結果
                timestamp = time.time() - start_time
                self._save_results(round_num, accuracy, loss, timestamp)
                
                # 每輪都更新性能圖表
                self._plot_performance()
                
                # 記錄 round 用時
                round_time = time.time() - round_start_time
                print(f"===== Round {round_num} 完成，用時 {round_time:.2f} 秒，準確率 {accuracy:.2f}% =====")
            
            # 釋放記憶體
            del broadcast_model
            self._clean_memory()
            
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
        print("停止 Spark 會話...")
        self.spark.stop()
        print("Spark 會話已停止")

def main():
    print("開始初始化Spark聯邦學習故障恢復實驗...")
    
    # 創建 SparkFL 實例並運行訓練
    spark = SparkSession.builder \
        .appName("SparkFL_FaultTolerance_Exp1") \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
        .config("spark.default.parallelism", "2") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.task.maxFailures", "1") \
        .config("spark.stage.maxConsecutiveAttempts", "2") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.network.timeout", "600s") \
        .config("spark.executor.heartbeatInterval", "20s") \
        .config("spark.rpc.askTimeout", "600s") \
        .config("spark.rpc.lookupTimeout", "600s") \
        .config("spark.sql.broadcastTimeout", "600") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()
    
    # 設置日誌級別
    spark.sparkContext.setLogLevel("WARN")
    
    print("Spark會話創建成功")
    
    spark_fl = SparkFL(spark, num_workers=2, num_rounds=20)
    try:
        spark_fl.train()
    except Exception as e:
        print(f"訓練過程中發生錯誤: {e}")
        traceback.print_exc()
    finally:
        spark_fl.stop()

if __name__ == "__main__":
    main() 