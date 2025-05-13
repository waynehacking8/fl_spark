#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import Row

def evaluate_model(model, test_loader, device='cpu'):
    """評估模型性能"""
    model.eval()
    correct = 0
    total = 0
    predictions = []
    labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            labels.extend(target.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, predictions, labels

def create_spark_metrics(predictions, labels):
    """創建 Spark 評估指標"""
    prediction_and_labels = [Row(prediction=float(p), label=float(l)) 
                           for p, l in zip(predictions, labels)]
    
    df = spark.createDataFrame(prediction_and_labels)
    metrics = MulticlassMetrics(df.rdd.map(lambda x: (x.prediction, x.label)))
    
    return {
        'accuracy': metrics.accuracy,
        'precision': metrics.precision(),
        'recall': metrics.recall(),
        'f1_score': metrics.fMeasure()
    }

def save_model(model, path):
    """保存模型"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """加載模型"""
    model.load_state_dict(torch.load(path))
    return model

def log_metrics(metrics, epoch):
    """記錄評估指標"""
    print(f"Epoch {epoch}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}") 