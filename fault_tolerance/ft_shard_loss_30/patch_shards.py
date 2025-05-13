#!/usr/bin/env python
import torch, os, random, sys

LOSS_RATE = 0.30
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

for i in range(1, 17):
    path = os.path.join(DATA_DIR, f'mnist_train_part{i}.pt')
    if not os.path.exists(path):
        continue
    buf = torch.load(path)
    data = buf['data']
    targets = buf['targets']
    n = len(data)
    keep = int(n * (1 - LOSS_RATE))
    idx = torch.randperm(n)[:keep]
    buf['data'] = data[idx]
    buf['targets'] = targets[idx]
    torch.save(buf, path)
print(f"✓ Shards patched with {LOSS_RATE*100:.0f}% loss (kept {keep}/{n})") 