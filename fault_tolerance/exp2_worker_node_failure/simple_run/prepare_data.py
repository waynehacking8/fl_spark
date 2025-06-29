#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pack MNIST into 4 shards for 4 participants, each with 15000 samples."""
from torchvision import datasets
import torch, os

DATA_DIR = './data'

os.makedirs(DATA_DIR, exist_ok=True)

print('Downloading / loading MNIST …')
train_ds = datasets.MNIST(DATA_DIR, train=True,  download=True)
test_ds  = datasets.MNIST(DATA_DIR, train=False, download=True)

# 切成 4 份，每份 15000 筆
num_parts = 4
samples_per_part = 15000

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