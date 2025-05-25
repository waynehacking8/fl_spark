#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIFAR-10 Data Preparation for Federated Learning
下載CIFAR-10數據集並分割為2個分片，每分片30,000樣本
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import logging
from torch.utils.data import DataLoader, TensorDataset

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_cifar10():
    """下載CIFAR-10數據集"""
    logging.info("正在下載CIFAR-10數據集...")
    
    # CIFAR-10標準化參數
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 下載訓練數據
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data/raw', 
        train=True, 
        download=True,
        transform=transform
    )
    
    # 下載測試數據
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data/raw', 
        train=False, 
        download=True,
        transform=transform
    )
    
    logging.info(f"✅ CIFAR-10數據集下載完成")
    logging.info(f"訓練樣本數: {len(train_dataset)}")
    logging.info(f"測試樣本數: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def create_federated_data(train_dataset, test_dataset, num_participants=2):
    """
    創建聯邦學習數據分片
    
    Args:
        train_dataset: CIFAR-10訓練數據集
        test_dataset: CIFAR-10測試數據集  
        num_participants: 參與者數量（默認2個）
    """
    logging.info(f"正在創建 {num_participants} 個聯邦學習數據分片...")
    
    # 轉換為tensor格式
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    # 獲取所有訓練數據
    train_data, train_targets = next(iter(train_loader))
    test_data, test_targets = next(iter(test_loader))
    
    logging.info(f"訓練數據形狀: {train_data.shape}")
    logging.info(f"測試數據形狀: {test_data.shape}")
    
    # 數據分片 - IID分配
    total_samples = len(train_data)
    samples_per_participant = total_samples // num_participants
    
    logging.info(f"總訓練樣本: {total_samples}")
    logging.info(f"每個參與者樣本數: {samples_per_participant}")
    
    # 隨機打亂索引以確保IID分配
    indices = np.random.permutation(total_samples)
    
    # 創建數據目錄
    os.makedirs('./data', exist_ok=True)
    
    # 分割並保存訓練數據
    for i in range(num_participants):
        start_idx = i * samples_per_participant
        end_idx = (i + 1) * samples_per_participant
        
        # 獲取當前參與者的數據索引
        participant_indices = indices[start_idx:end_idx]
        
        # 提取對應的數據和標籤
        participant_data = train_data[participant_indices]
        participant_targets = train_targets[participant_indices]
        
        # 保存分片數據
        shard_file = f'./data/cifar10_train_part{i+1}.pt'
        torch.save({
            'data': participant_data,
            'targets': participant_targets
        }, shard_file)
        
        logging.info(f"✅ 參與者 {i+1} 數據已保存: {shard_file}")
        logging.info(f"   樣本數: {len(participant_data)}")
        logging.info(f"   數據形狀: {participant_data.shape}")
        logging.info(f"   類別分布: {torch.bincount(participant_targets)}")
    
    # 保存測試數據（所有參與者共享）
    test_file = './data/cifar10_test.pt'
    torch.save({
        'data': test_data,
        'targets': test_targets
    }, test_file)
    
    logging.info(f"✅ 測試數據已保存: {test_file}")
    logging.info(f"   測試樣本數: {len(test_data)}")

def verify_data_integrity():
    """驗證數據完整性"""
    logging.info("正在驗證數據完整性...")
    
    # 檢查分片文件
    total_samples = 0
    for i in range(1, 3):  # 2個參與者
        shard_file = f'./data/cifar10_train_part{i}.pt'
        if os.path.exists(shard_file):
            data = torch.load(shard_file)
            samples = len(data['data'])
            total_samples += samples
            logging.info(f"✅ 分片 {i}: {samples} 樣本")
        else:
            logging.error(f"❌ 分片文件不存在: {shard_file}")
            return False
    
    # 檢查測試文件
    test_file = './data/cifar10_test.pt'
    if os.path.exists(test_file):
        test_data = torch.load(test_file)
        test_samples = len(test_data['data'])
        logging.info(f"✅ 測試數據: {test_samples} 樣本")
    else:
        logging.error(f"❌ 測試文件不存在: {test_file}")
        return False
    
    logging.info(f"✅ 數據完整性驗證通過")
    logging.info(f"總訓練樣本: {total_samples}")
    logging.info(f"測試樣本: {test_samples}")
    
    return True

def display_sample_images():
    """顯示樣本圖像信息"""
    logging.info("正在分析樣本數據...")
    
    # CIFAR-10類別名稱
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 加載第一個分片查看數據
    shard_file = './data/cifar10_train_part1.pt'
    if os.path.exists(shard_file):
        data = torch.load(shard_file)
        sample_data = data['data']
        sample_targets = data['targets']
        
        logging.info(f"📊 分片1數據分析:")
        logging.info(f"   數據類型: {sample_data.dtype}")
        logging.info(f"   數據範圍: [{sample_data.min():.3f}, {sample_data.max():.3f}]")
        logging.info(f"   圖像尺寸: {sample_data.shape[1:]}") # [C, H, W]
        
        # 類別分布統計
        class_counts = torch.bincount(sample_targets)
        logging.info(f"   類別分布:")
        for i, count in enumerate(class_counts):
            logging.info(f"     {class_names[i]}: {count} 樣本")

def main():
    """主函數"""
    print("🎯 CIFAR-10 聯邦學習數據準備")
    print("=" * 50)
    
    try:
        # 1. 下載CIFAR-10數據集
        train_dataset, test_dataset = download_cifar10()
        
        # 2. 創建聯邦學習數據分片
        create_federated_data(train_dataset, test_dataset, num_participants=2)
        
        # 3. 驗證數據完整性
        if verify_data_integrity():
            logging.info("🎉 CIFAR-10數據準備完成！")
        else:
            logging.error("❌ 數據驗證失敗！")
            return
        
        # 4. 顯示樣本信息
        display_sample_images()
        
        print("\n📁 生成的文件:")
        print("  - ./data/cifar10_train_part1.pt (參與者1訓練數據)")
        print("  - ./data/cifar10_train_part2.pt (參與者2訓練數據)")
        print("  - ./data/cifar10_test.pt (共享測試數據)")
        print("  - ./data/raw/ (原始CIFAR-10數據)")
        
    except Exception as e:
        logging.error(f"❌ 數據準備失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 