#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from torchvision import datasets, transforms
import torch

class SparkDataLoader:
    def __init__(self, spark_session, num_partitions=10):
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self.num_partitions = num_partitions
    
    def load_mnist(self, data_dir='../data/mnist/', iid=True):
        """加載 MNIST 數據集並轉換為 Spark RDD"""
        # 加載 MNIST 數據
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        
        # 將數據轉換為 Spark RDD
        data_points = []
        for i in range(len(dataset)):
            image, label = dataset[i]
            data_points.append((i, (image.numpy(), label)))
        
        # 創建 RDD
        rdd = self.sc.parallelize(data_points, self.num_partitions)
        
        if not iid:
            # 實現 Non-IID 數據分佈
            rdd = self._create_noniid_distribution(rdd)
        
        return rdd
    
    def _create_noniid_distribution(self, rdd):
        """創建 Non-IID 數據分佈"""
        # 按標籤分組
        labeled_data = rdd.map(lambda x: (x[1][1], x))
        
        # 將數據按標籤排序
        sorted_data = labeled_data.sortByKey()
        
        # 重新分區以創建 Non-IID 分佈
        def partitioner(key):
            return key % self.num_partitions
        
        return sorted_data.map(lambda x: x[1]).partitionBy(self.num_partitions, partitioner)
    
    def create_test_loader(self, data_dir='../data/mnist/'):
        """創建測試數據加載器"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        return torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True) 