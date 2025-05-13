#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from pyspark.sql import SparkSession
import numpy as np
from typing import Dict, Any
import json
from .differential_privacy import DifferentialPrivacy, DPConfig
from .participant_selection import DynamicParticipantSelector

class FederatedStreaming:
    def __init__(self, spark_session: SparkSession, batch_interval: int = 10):
        """
        初始化聯邦學習流處理
        
        Args:
            spark_session: SparkSession 實例
            batch_interval: 批次處理間隔（秒）
        """
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self.ssc = StreamingContext(self.sc, batch_interval)
        
        # 初始化差分隱私
        self.dp = DifferentialPrivacy(DPConfig(
            epsilon=1.0,
            delta=1e-5,
            sensitivity=1.0,
            clip_norm=1.0
        ))
        
        # 初始化參與者選擇器
        self.selector = DynamicParticipantSelector(spark_session)
        
        # 創建 Kafka 流
        self.kafka_stream = KafkaUtils.createStream(
            self.ssc,
            "zookeeper:2181",
            "federated_learning_group",
            {"mnist_data": 1}
        )
        
    def parse_data(self, message: str) -> Dict[str, Any]:
        """解析 Kafka 消息"""
        try:
            data = json.loads(message)
            return {
                'participant_id': data['participant_id'],
                'features': np.array(data['features']),
                'label': data['label']
            }
        except Exception as e:
            print(f"解析數據錯誤: {e}")
            return None
            
    def train_batch(self, rdd):
        """處理每個批次的數據"""
        if rdd.isEmpty():
            return
            
        # 轉換為 DataFrame
        df = self.spark.createDataFrame(rdd)
        
        # 更新參與者指標
        metrics = df.groupBy('participant_id').applyInPandas(
            lambda pdf: pd.DataFrame([self.selector.update_metrics(
                pdf['participant_id'].iloc[0],
                pdf
            )]),
            "participant_id int, cpu_percent double, memory_percent double, "
            "disk_usage double, data_quality_score double"
        )
        
        # 選擇參與者
        selected_participants = self.selector.select_participants(metrics)
        
        # 過濾選中的參與者數據
        selected_data = df.filter(df.participant_id.isin(selected_participants))
        
        # 訓練模型
        model_updates = selected_data.groupBy('participant_id').applyInPandas(
            self._train_local_model,
            "participant_id int, weights array<double>"
        )
        
        # 應用差分隱私
        private_updates = model_updates.rdd.map(
            lambda row: (row.participant_id, self.dp.add_noise(np.array(row.weights)))
        )
        
        # 聚合更新
        aggregated_weights = private_updates.map(
            lambda x: x[1]
        ).reduce(lambda x, y: x + y) / len(selected_participants)
        
        # 更新全局模型
        self._update_global_model(aggregated_weights)
        
        # 記錄指標
        self._log_metrics(selected_participants, aggregated_weights)
        
    def _train_local_model(self, pdf):
        """訓練本地模型"""
        from sklearn.linear_model import LogisticRegression
        
        X = np.array(pdf['features'].tolist())
        y = np.array(pdf['label'].tolist())
        
        model = LogisticRegression(max_iter=1)
        model.fit(X, y)
        
        return pd.DataFrame([{
            'participant_id': pdf['participant_id'].iloc[0],
            'weights': model.coef_.flatten().tolist()
        }])
        
    def _update_global_model(self, weights: np.ndarray):
        """更新全局模型"""
        # 這裡可以實現模型更新邏輯
        # 例如：保存到文件或廣播到所有節點
        pass
        
    def _log_metrics(self, selected_participants: list, weights: np.ndarray):
        """記錄指標"""
        metrics = {
            'selected_participants': len(selected_participants),
            'privacy_spent': self.dp.privacy_spent,
            'weights_norm': np.linalg.norm(weights)
        }
        
        # 將指標發送到 Prometheus
        self._send_metrics_to_prometheus(metrics)
        
    def _send_metrics_to_prometheus(self, metrics: Dict[str, Any]):
        """發送指標到 Prometheus"""
        from prometheus_client import Counter, Gauge
        import prometheus_client
        
        # 定義指標
        participants_gauge = Gauge('fl_selected_participants', 'Number of selected participants')
        privacy_gauge = Gauge('fl_privacy_spent', 'Privacy budget spent')
        weights_gauge = Gauge('fl_weights_norm', 'Norm of model weights')
        
        # 更新指標
        participants_gauge.set(metrics['selected_participants'])
        privacy_gauge.set(metrics['privacy_spent'])
        weights_gauge.set(metrics['weights_norm'])
        
    def start(self):
        """啟動流處理"""
        # 處理數據流
        self.kafka_stream.map(lambda x: x[1]).map(self.parse_data).foreachRDD(self.train_batch)
        
        # 啟動流處理
        self.ssc.start()
        self.ssc.awaitTermination()
        
    def stop(self):
        """停止流處理"""
        self.ssc.stop(stopSparkContext=False, stopGraceFully=True) 