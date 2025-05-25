#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIFAR-10 CNN Model Definitions for Federated Learning
適配CIFAR-10的CNN模型架構
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10CNN(nn.Module):
    """
    適配CIFAR-10的CNN模型
    輸入: 32x32x3 彩色圖像
    輸出: 10個類別的分類結果
    """
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        
        # 第一個卷積塊
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32x32x32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # 32x32x32
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 16x16x32
        self.dropout1 = nn.Dropout2d(0.25)
        
        # 第二個卷積塊
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 16x16x64
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 16x16x64
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 8x8x64
        self.dropout2 = nn.Dropout2d(0.25)
        
        # 第三個卷積塊
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 8x8x128
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 8x8x128
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 4x4x128
        self.dropout3 = nn.Dropout2d(0.25)
        
        # 全連接層
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn8 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # 第一個卷積塊
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 第二個卷積塊
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 第三個卷積塊
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # 展平並通過全連接層
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout4(x)
        x = F.relu(self.bn8(self.fc2(x)))
        x = self.dropout5(x)
        x = self.fc3(x)
        
        return x

class CIFAR10CNNSimple(nn.Module):
    """
    簡化版CIFAR-10 CNN模型（用於快速測試）
    """
    def __init__(self, num_classes=10):
        super(CIFAR10CNNSimple, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ResidualBlock(nn.Module):
    """殘差塊（用於更深的網絡）"""
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
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.dropout(out)
        return out

class CIFAR10ResNet(nn.Module):
    """
    ResNet風格的CIFAR-10模型（用於高性能需求）
    """
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(CIFAR10ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(dropout_rate * 0.5)  # 卷積層較低dropout
        
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(x)
        x = self.fc(x)
        return x

def get_model(model_type='standard', num_classes=10):
    """
    獲取指定類型的模型
    
    Args:
        model_type: 模型類型 ('standard', 'simple', 'resnet')
        num_classes: 類別數量
    
    Returns:
        PyTorch模型實例
    """
    if model_type == 'standard':
        return CIFAR10CNN(num_classes)
    elif model_type == 'simple':
        return CIFAR10CNNSimple(num_classes)
    elif model_type == 'resnet':
        return CIFAR10ResNet(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model):
    """計算模型參數數量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def test_models():
    """測試所有模型架構"""
    print("🧪 測試CIFAR-10模型架構")
    print("=" * 50)
    
    # 創建測試數據
    test_input = torch.randn(1, 3, 32, 32)
    
    models = {
        'Standard CNN': CIFAR10CNN(),
        'Simple CNN': CIFAR10CNNSimple(),
        'ResNet': CIFAR10ResNet()
    }
    
    for name, model in models.items():
        print(f"\n📊 {name}:")
        
        # 測試前向傳播
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        # 計算參數
        total_params, trainable_params = count_parameters(model)
        
        print(f"  輸入形狀: {test_input.shape}")
        print(f"  輸出形狀: {output.shape}")
        print(f"  總參數數: {total_params:,}")
        print(f"  可訓練參數: {trainable_params:,}")
        print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    test_models() 