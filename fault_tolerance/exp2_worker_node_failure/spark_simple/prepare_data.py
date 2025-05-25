#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
數據準備腳本 - Spark FL 簡化版本
"""
import torch
import os
from torchvision import datasets, transforms

def prepare_mnist_data():
    """準備MNIST數據分片"""
    print("準備 MNIST 數據...")
    
    # 創建數據目錄
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # 數據轉換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 下載MNIST數據
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    # 分割訓練數據為4個參與者
    total_samples = len(train_dataset)
    samples_per_participant = total_samples // 4
    
    print(f"總訓練樣本: {total_samples}")
    print(f"每個參與者樣本數: {samples_per_participant}")
    
    for i in range(4):
        start_idx = i * samples_per_participant
        end_idx = start_idx + samples_per_participant
        
        # 提取數據和標籤
        participant_data = []
        participant_targets = []
        
        for idx in range(start_idx, end_idx):
            data, target = train_dataset[idx]
            participant_data.append(data)
            participant_targets.append(target)
        
        # 堆疊為張量
        participant_data = torch.stack(participant_data)
        participant_targets = torch.tensor(participant_targets)
        
        # 保存分片
        participant_file = os.path.join(data_dir, f'participant_{i+1}_data.pt')
        torch.save({
            'data': participant_data,
            'targets': participant_targets
        }, participant_file)
        
        print(f"參與者 {i+1} 數據已保存: {participant_file} ({len(participant_data)} 樣本)")
    
    # 保存測試數據
    test_data = []
    test_targets = []
    for data, target in test_dataset:
        test_data.append(data)
        test_targets.append(target)
    
    test_data = torch.stack(test_data)
    test_targets = torch.tensor(test_targets)
    
    test_file = os.path.join(data_dir, 'test_data.pt')
    torch.save({
        'data': test_data,
        'targets': test_targets
    }, test_file)
    
    print(f"測試數據已保存: {test_file} ({len(test_data)} 樣本)")
    print("數據準備完成！")

if __name__ == "__main__":
    prepare_mnist_data() 