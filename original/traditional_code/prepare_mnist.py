#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pack MNIST into a single .pt file to avoid every participant re-extracting the gzip files.
Runs once inside the data-init container and writes mnist_train.pt / mnist_test.pt to /app/data.
"""
from torchvision import datasets
import torch, os

DATA_DIR = '/app/data'

os.makedirs(DATA_DIR, exist_ok=True)

print('Downloading / loading MNIST …')
train_ds = datasets.MNIST(DATA_DIR, train=True,  download=True)
test_ds  = datasets.MNIST(DATA_DIR, train=False, download=True)

# 切成 16 份，每份 3750 筆，直接存為 uint8 tensor（節省空間）
num_parts = 16
samples_per_part = len(train_ds.data) // num_parts

print('Saving per-participant shards …')
for i in range(num_parts):
    s = i * samples_per_part
    e = s + samples_per_part
    shard = {
        'data': train_ds.data[s:e],      # uint8 28x28
        'targets': train_ds.targets[s:e]
    }
    torch.save(shard, os.path.join(DATA_DIR, f'mnist_train_part{i+1}.pt'))

# 測試集保持完整一份
torch.save({'data': test_ds.data, 'targets': test_ds.targets}, os.path.join(DATA_DIR, 'mnist_test.pt'))

print('✓ Shards written') 