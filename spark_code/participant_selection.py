#!/usr/bin/env python
# -*- coding: utf-8 -*-

import psutil
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand

class DynamicParticipantSelector:
    def __init__(self, spark_session: SparkSession, selection_ratio=0.5):
        """
        初始化動態參與者選擇器
        
        Args:
            spark_session: SparkSession 實例
            selection_ratio: 每輪選擇的參與者比例
        """
        self.spark = spark_session
        self.selection_ratio = selection_ratio
        
    def collect_resource_metrics(self):
        """收集資源使用指標"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
        return metrics
        
    def evaluate_data_quality(self, participant_data):
        """評估參與者數據質量"""
        # 計算數據分佈熵
        label_counts = (participant_data.select('label')
                       .groupBy('label')
                       .count()
                       .collect())
        
        total = sum(row['count'] for row in label_counts)
        probabilities = [row['count']/total for row in label_counts]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        return entropy
        
    def select_participants(self, participant_metrics_df):
        """
        基於資源指標和數據質量選擇參與者
        
        Args:
            participant_metrics_df: 包含參與者指標的DataFrame
            
        Returns:
            選中的參與者ID列表
        """
        # 計算綜合分數
        scored_df = participant_metrics_df.withColumn(
            'score',
            (1 - col('cpu_percent')/100) * 0.4 +
            (1 - col('memory_percent')/100) * 0.3 +
            col('data_quality_score') * 0.3
        )
        
        # 根據分數和隨機因素選擇參與者
        n_participants = participant_metrics_df.count()
        n_select = max(2, int(n_participants * self.selection_ratio))
        
        selected = (scored_df
                   .orderBy(col('score').desc(), rand())
                   .limit(n_select)
                   .select('participant_id')
                   .collect())
        
        return [row['participant_id'] for row in selected]
    
    def update_metrics(self, participant_id, data):
        """
        更新參與者的指標
        
        Args:
            participant_id: 參與者ID
            data: 參與者的訓練數據
        """
        metrics = self.collect_resource_metrics()
        data_quality = self.evaluate_data_quality(data)
        
        return {
            'participant_id': participant_id,
            'cpu_percent': metrics['cpu_percent'],
            'memory_percent': metrics['memory_percent'],
            'disk_usage': metrics['disk_usage'],
            'data_quality_score': data_quality
        } 