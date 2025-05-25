#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spark FL Implementation for CIFAR-10 with RDD Lineage Fault Tolerance (Improved)
基於Spark的CIFAR-10聯邦學習實現，支持RDD血統追蹤容錯（改進版）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import logging
import os
import sys
from pyspark.sql import SparkSession
from pyspark.broadcast import Broadcast
import pickle

# 添加父目錄到路徑以導入統一模型
sys.path.append('..')
from models import CIFAR10ResNet, CIFAR10CNNSimple

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - CIFAR10-SparkFL-Improved - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def precise_federated_averaging(model_updates, weights, reference_model):
    """
    精確的聯邦平均算法，與傳統FL保持一致
    """
    if not model_updates:
        logging.warning("沒有模型更新，跳過聚合")
        return None
    
    # 計算歸一化權重
    total_samples = sum(weights)
    normalized_weights = [w / total_samples for w in weights]
    
    # 獲取參考模型的狀態字典作為模板
    reference_state = reference_model.state_dict()
    global_state_dict = {}
    
    # 獲取設備信息
    device = next(reference_model.parameters()).device
    
    # 初始化 - 保持數據類型
    for key in model_updates[0].keys():
        global_state_dict[key] = torch.zeros_like(reference_state[key]).to(device)
    
    # 精確加權平均 - 先轉為float計算再轉回原類型
    for i, update in enumerate(model_updates):
        weight = normalized_weights[i]
        for key in update.keys():
            # 獲取原始數據類型
            original_dtype = global_state_dict[key].dtype
            param_tensor = update[key].to(device)  # 確保在正確設備上
            
            # 轉為float進行精確計算
            if param_tensor.dtype != torch.float32:
                param_tensor = param_tensor.float()
            if global_state_dict[key].dtype != torch.float32:
                global_state_dict[key] = global_state_dict[key].float()
            
            # 執行加權求和
            global_state_dict[key] += weight * param_tensor
            
            # 轉回原始類型
            if original_dtype != torch.float32:
                global_state_dict[key] = global_state_dict[key].to(original_dtype)
    
    logging.info(f"精確聯邦平均完成，使用 {len(model_updates)} 個更新")
    return global_state_dict

def train_federated_learning_spark(num_rounds=20, num_partitions=2, model_type='resnet', 
                                  batch_size=32, learning_rate=0.001, local_epochs=5,
                                  experiment_mode='normal'):
    """
    使用Apache Spark進行聯邦學習訓練 (CIFAR-10) - 改進版
    """
    # 實驗模式配置
    if experiment_mode == 'normal':
        fault_round = None
        failed_partition = None
        print("🔄 正常模式：無故障注入")
    elif experiment_mode == 'exp1':
        fault_round = 5
        failed_partition = 0
        print("🧪 實驗1模式：第5輪數據分片貢獻失敗 (partition 0)")
    elif experiment_mode == 'exp2':
        fault_round = 8
        failed_partition = 0
        print("🔧 實驗2模式：第8輪Worker節點故障 (partition 0)")
    else:
        raise ValueError(f"Unknown experiment mode: {experiment_mode}")
    
    # 設備配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    # 檢查數據文件
    train_data_files = [
        "../data/cifar10_train_part1.pt",
        "../data/cifar10_train_part2.pt"
    ]
    test_data_file = "../data/cifar10_test.pt"
    
    for file_path in train_data_files + [test_data_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"數據文件不存在: {file_path}")
    
    print("✅ 數據文件檢查完成")
    
    # 初始化Spark
    spark = SparkSession.builder \
        .appName(f"CIFAR10FederatedLearning_Improved_{experiment_mode}") \
        .master("local[2]") \
        .config("spark.driver.memory", "12g") \
        .config("spark.executor.memory", "12g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.python.worker.memory", "8g") \
        .config("spark.driver.memoryFraction", "0.8") \
        .config("spark.executor.memoryFraction", "0.8") \
        .config("spark.storage.memoryFraction", "0.6") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryo.unsafe", "true") \
        .config("spark.kryo.maxCacheSize", "1g") \
        .config("spark.kryoserializer.buffer.max", "1g") \
        .config("spark.kryoserializer.buffer", "64m") \
        .config("spark.rdd.compress", "true") \
        .config("spark.broadcast.compress", "true") \
        .config("spark.io.compression.codec", "lz4") \
        .config("spark.default.parallelism", "2") \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
        .config("spark.python.worker.reuse", "false") \
        .config("spark.executor.cores", "2") \
        .config("spark.cores.max", "2") \
        .config("spark.executor.memoryStorageLevel", "MEMORY_ONLY") \
        .config("spark.serializer.objectStreamReset", "100") \
        .getOrCreate()
    
    sc = spark.sparkContext
    sc.setLogLevel("WARN")
    
    print(f"🚀 Spark 初始化完成 ({experiment_mode}模式) - 改進版")
    
    # 創建結果目錄
    results_dir = f"../results/spark/{experiment_mode}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化結果文件 - 直接覆蓋原文件
    results_file = os.path.join(results_dir, f'cifar10_spark_{experiment_mode}_results.csv')
    with open(results_file, 'w') as f:
        f.write("Round,Timestamp,Accuracy,Loss,Active_Partitions,Failed_Partitions,Mode\n")
    
    # 初始化全局模型 - 使用統一模型定義
    if model_type == 'simple':
        global_model = CIFAR10CNNSimple(num_classes=10).to(device)
    else:
        global_model = CIFAR10ResNet(num_classes=10, dropout_rate=0.3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # 加載測試數據
    test_data = torch.load(test_data_file, weights_only=False)
    test_dataset = TensorDataset(test_data['data'], test_data['targets'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 創建分布式數據RDD（分批加載以減少內存壓力）
    partition_info = []
    for i, file_path in enumerate(train_data_files):
        if i < num_partitions:
            # 只保存文件路徑，不直接加載大數據
            partition_info.append((i, file_path))
    
    # 創建較小的RDD，僅包含分區信息
    data_rdd = sc.parallelize(partition_info, num_partitions)
    data_rdd.cache()  # 啟用緩存
    
    print(f"📊 創建 {num_partitions} 個分區的分布式數據集（懶加載模式）")
    
    # 記錄總開始時間
    total_start_time = time.time()
    
    # 聯邦學習主循環
    for round_num in range(1, num_rounds + 1):
        print(f"\n=== 第 {round_num} 輪開始 ({experiment_mode}模式) - 改進版 ===")
        round_start_time = time.time()
        
        # 廣播全局模型參數
        model_state = global_model.state_dict()
        broadcasted_model = sc.broadcast(model_state)
        
        # 故障注入
        if fault_round is not None and round_num == fault_round and failed_partition is not None:
            if experiment_mode == 'exp1':
                print(f"🧪 第 {round_num} 輪數據分片貢獻失敗：partition {failed_partition} 故障")
            elif experiment_mode == 'exp2':
                print(f"🔧 第 {round_num} 輪Worker節點故障：partition {failed_partition} 故障")
            
            # 過濾故障分區
            active_rdd = data_rdd.filter(lambda x: x[0] != failed_partition)
        else:
            active_rdd = data_rdd
        
        # 分布式訓練函數 - 改進版
        def train_partition_improved(partition_info):
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            import numpy as np
            import sys
            import os
            
            # 添加父目錄到路徑
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            
            # 導入統一模型定義
            try:
                from models import CIFAR10ResNet, CIFAR10CNNSimple
            except ImportError:
                # 如果導入失敗，使用內聯定義作為備份
                class ResidualBlock(nn.Module):
                    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1):
                        super(ResidualBlock, self).__init__()
                        
                        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
                        self.bn1 = nn.BatchNorm2d(out_channels)
                        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
                        self.bn2 = nn.BatchNorm2d(out_channels)
                        
                        self.shortcut = nn.Sequential()
                        if stride != 1 or in_channels != out_channels:
                            self.shortcut = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                                nn.BatchNorm2d(out_channels)
                            )
                        
                        self.dropout = nn.Dropout2d(dropout_rate)
                    
                    def forward(self, x):
                        out = torch.relu(self.bn1(self.conv1(x)))
                        out = self.bn2(self.conv2(out))
                        out += self.shortcut(x)
                        out = torch.relu(out)
                        out = self.dropout(out)
                        return out

                class CIFAR10ResNet(nn.Module):
                    def __init__(self, num_classes=10, dropout_rate=0.3):
                        super(CIFAR10ResNet, self).__init__()
                        
                        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
                        self.bn1 = nn.BatchNorm2d(32)
                        self.dropout1 = nn.Dropout2d(dropout_rate * 0.5)
                        
                        self.layer1 = self._make_layer(32, 32, 2, stride=1, dropout_rate=dropout_rate * 0.5)
                        self.layer2 = self._make_layer(32, 64, 2, stride=2, dropout_rate=dropout_rate * 0.7)
                        self.layer3 = self._make_layer(64, 128, 2, stride=2, dropout_rate=dropout_rate)
                        
                        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                        self.dropout_fc = nn.Dropout(dropout_rate)
                        self.fc = nn.Linear(128, num_classes)
                        
                    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_rate=0.1):
                        layers = []
                        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout_rate))
                        for _ in range(1, num_blocks):
                            layers.append(ResidualBlock(out_channels, out_channels, dropout_rate=dropout_rate))
                        return nn.Sequential(*layers)
                    
                    def forward(self, x):
                        x = torch.relu(self.bn1(self.conv1(x)))
                        x = self.dropout1(x)
                        x = self.layer1(x)
                        x = self.layer2(x)
                        x = self.layer3(x)
                        x = self.avg_pool(x)
                        x = x.view(x.size(0), -1)
                        x = self.dropout_fc(x)
                        x = self.fc(x)
                        return x
            
            partition_id, file_path = partition_info
            
            try:
                # 檢查分區是否在故障列表中
                if (fault_round is not None and round_num == fault_round and 
                    failed_partition is not None and partition_id == failed_partition):
                    if experiment_mode == 'exp1':
                        print(f"💥 Partition {partition_id} 模擬數據分片貢獻失敗")
                    elif experiment_mode == 'exp2':
                        print(f"💥 Partition {partition_id} 模擬Worker節點故障")
                    raise RuntimeError(f"Simulated partition {partition_id} failure")
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # 創建本地模型 - 使用統一定義
                if model_type == 'simple':
                    local_model = CIFAR10CNNSimple(num_classes=10).to(device)
                else:
                    local_model = CIFAR10ResNet(num_classes=10, dropout_rate=0.3).to(device)
                
                local_model.load_state_dict(broadcasted_model.value)
                local_model.train()
                
                # 準備數據
                data = torch.load(file_path, weights_only=False)
                dataset = TensorDataset(data['data'], data['targets'])
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                # 改進的優化器配置 - 添加L2正則化
                weight_decay = 1e-4  # L2正則化
                optimizer = optim.Adam(local_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                
                # 學習率調度器 - 每5輪減少學習率
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
                
                criterion = nn.CrossEntropyLoss()
                
                # 本地訓練
                local_losses = []
                for epoch in range(local_epochs):
                    epoch_losses = []
                    for batch_data, batch_targets in data_loader:
                        batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
                        
                        optimizer.zero_grad()
                        outputs = local_model(batch_data)
                        loss = criterion(outputs, batch_targets)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_losses.append(loss.item())
                    
                    local_losses.append(np.mean(epoch_losses))
                
                # 更新學習率調度器
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                
                # 返回模型參數和訓練信息 - 確保數據類型一致性
                model_state_cpu = {}
                for key, value in local_model.state_dict().items():
                    model_state_cpu[key] = value.cpu().clone()
                
                return {
                    'partition_id': partition_id,
                    'model_state': model_state_cpu,
                    'data_size': len(dataset),
                    'avg_loss': np.mean(local_losses),
                    'current_lr': current_lr,
                    'status': 'success'
                }
                
            except Exception as e:
                # 模擬故障恢復
                print(f"⚠️  Partition {partition_id} 訓練失敗: {str(e)}")
                return {
                    'partition_id': partition_id,
                    'status': 'failed',
                    'error': str(e)
                }
        
        # 執行分布式訓練並收集結果
        try:
            training_results = active_rdd.map(train_partition_improved).collect()
        except Exception as e:
            print(f"❌ 分布式訓練失敗: {e}")
            # 利用RDD lineage進行故障恢復
            print("🔄 使用RDD lineage進行故障恢復...")
            training_results = active_rdd.map(train_partition_improved).collect()
        
        # 過濾成功的結果
        successful_results = [r for r in training_results if r['status'] == 'success']
        failed_results = [r for r in training_results if r['status'] == 'failed']
        
        if failed_results:
            print(f"⚠️  {len(failed_results)} 個分區訓練失敗，使用 {len(successful_results)} 個成功分區進行聚合")
        
        if not successful_results:
            print(f"❌ 第 {round_num} 輪沒有成功的分區，跳過聚合")
            continue
        
        # 使用精確聯邦平均算法
        model_updates = [r['model_state'] for r in successful_results]
        weights = [r['data_size'] for r in successful_results]
        
        new_state_dict = precise_federated_averaging(model_updates, weights, global_model)
        
        if new_state_dict is not None:
            # 更新全局模型
            global_model.load_state_dict(new_state_dict)
        else:
            print(f"❌ 第 {round_num} 輪聯邦平均失敗")
            continue
        
        # 評估模型
        global_model.eval()
        correct = 0
        total = 0
        total_loss = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = global_model(data)
                loss = criterion(outputs, target)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(test_loader)
        cumulative_time = time.time() - total_start_time  # 使用累計時間
        
        # 記錄結果
        active_partitions = len(successful_results)
        failed_partitions = len(failed_results)
        
        with open(results_file, 'a') as f:
            f.write(f"{round_num},{cumulative_time:.2f},{accuracy:.2f},{avg_loss:.4f},"
                   f"{active_partitions},{failed_partitions},{experiment_mode}\n")
        
        # 保存模型檢查點
        checkpoint_path = os.path.join(results_dir, f'model_round_{round_num}.pth')
        torch.save(global_model.state_dict(), checkpoint_path)
        
        print(f"第 {round_num} 輪完成:")
        print(f"  準確率: {accuracy:.2f}%")
        print(f"  損失: {avg_loss:.4f}")
        print(f"  累計用時: {cumulative_time:.2f}秒")
        print(f"  活躍分區: {active_partitions}/{num_partitions}")
        if failed_partitions > 0:
            print(f"  故障分區: {failed_partitions}")
        if successful_results:
            avg_lr = np.mean([r.get('current_lr', learning_rate) for r in successful_results])
            print(f"  當前學習率: {avg_lr:.6f}")
        
        # 清理廣播變量
        broadcasted_model.unpersist()
    
    # 清理資源
    data_rdd.unpersist()
    spark.stop()
    
    print("\n🎉 CIFAR-10 Spark聯邦學習完成！(改進版)")
    print(f"結果保存在: {results_file}")

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CIFAR-10 Federated Learning with Apache Spark')
    parser.add_argument('--rounds', type=int, default=20, help='Number of federated rounds')
    parser.add_argument('--partitions', type=int, default=2, help='Number of data partitions')
    parser.add_argument('--model', type=str, default='resnet', choices=['simple', 'resnet'],
                       help='Model type')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Local epochs')
    parser.add_argument('--mode', type=str, default='normal', choices=['normal', 'exp1', 'exp2'],
                       help='Experiment mode: normal, exp1 (data shard failure), exp2 (worker failure)')
    
    args = parser.parse_args()
    
    train_federated_learning_spark(
        num_rounds=args.rounds,
        num_partitions=args.partitions,
        model_type=args.model,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        local_epochs=args.epochs,
        experiment_mode=args.mode
    )

if __name__ == "__main__":
    main() 