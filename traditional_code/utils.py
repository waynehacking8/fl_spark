#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

def mnist_iid(dataset, num_users):
    """將 MNIST 數據集分割為 IID 分佈"""
    num_items = int(len(dataset)/num_users)
    dict_users = {}
    all_idxs = [i for i in range(len(dataset))]
    
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    
    return dict_users

def mnist_noniid(dataset, num_users):
    """將 MNIST 數據集分割為 Non-IID 分佈"""
    num_shards = num_users * 2
    num_imgs = int(len(dataset)/num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()
    
    # 按標籤排序
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    
    # 分割數據
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    
    return dict_users

def get_data_loader(dataset, idxs, batch_size=64):
    """獲取指定索引的數據加載器"""
    subset = Subset(dataset, list(idxs))
    return DataLoader(subset, batch_size=batch_size, shuffle=True)

def test_model(model, test_loader, device='cpu'):
    """測試模型性能"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy 