#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
from torchvision import datasets, transforms
import numpy as np

def prepare_mnist_data():
    """準備預處理的MNIST數據文件"""
    
    # 確保數據目錄存在
    data_dir = '/app/data'
    os.makedirs(data_dir, exist_ok=True)
    
    print("加載 MNIST 訓練數據...")
    
    # 使用原始數據轉換（不使用 transform）
    dataset = datasets.MNIST(data_dir, train=True, download=True)
    
    # 獲取所有數據
    data = dataset.data  # 形狀: [60000, 28, 28]
    targets = dataset.targets  # 形狀: [60000]
    
    print(f"數據形狀: {data.shape}, 標籤形狀: {targets.shape}")
    
    # 保存完整數據集到單一文件
    complete_data = {
        'data': data,
        'targets': targets
    }
    torch.save(complete_data, os.path.join(data_dir, 'mnist_train_complete.pt'))
    print(f"已保存 mnist_train_complete.pt，樣本數: {len(data)}")
    
    print("數據準備完成！")
    
    # 驗證文件
    print("\n驗證文件...")
    loaded_data = torch.load(os.path.join(data_dir, 'mnist_train_complete.pt'))
    
    print(f"Complete - 數據形狀: {loaded_data['data'].shape}, 標籤形狀: {loaded_data['targets'].shape}")

if __name__ == "__main__":
    prepare_mnist_data() 