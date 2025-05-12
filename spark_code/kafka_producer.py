#!/usr/bin/env python
# -*- coding: utf-8 -*-

from kafka import KafkaProducer
import numpy as np
import json
import time
from typing import Dict, Any
import random
from sklearn.datasets import fetch_openml
import logging
import sys
import pandas as pd

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MNISTDataProducer:
    def __init__(self, bootstrap_servers: str = 'kafka:9092', topic: str = 'mnist_data'):
        """
        初始化 MNIST 數據生產者
        
        Args:
            bootstrap_servers: Kafka 服務器地址
            topic: Kafka 主題名稱
        """
        self.logger = logger
        try:
            self.logger.info(f"正在連接到 Kafka 服務器: {bootstrap_servers}")
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                api_version=(2, 0, 2),
                retries=5,
                retry_backoff_ms=1000
            )
            self.topic = topic
            
            # 加載 MNIST 數據集
            self.logger.info("正在加載 MNIST 數據集...")
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
            self.X = X.astype(np.float32) / 255.0  # 歸一化
            self.y = y.astype(np.int32)
            self.logger.info(f"成功加載 MNIST 數據集，樣本數量: {len(self.X)}")
            
            self.logger.info(f"成功連接到 Kafka 服務器: {bootstrap_servers}")
            
        except Exception as e:
            self.logger.error(f"初始化 Kafka 生產者失敗: {str(e)}")
            sys.exit(1)
            
    def _create_participant_data(self, participant_id: int, num_samples: int = 100) -> Dict[str, Any]:
        """
        為參與者創建數據樣本
        
        Args:
            participant_id: 參與者 ID
            num_samples: 每個參與者的樣本數量
        """
        try:
            # 隨機選擇樣本
            indices = np.random.choice(len(self.X), num_samples, replace=False)
            X_participant = self.X[indices]
            y_participant = self.y[indices]
            
            # 確保數據格式正確
            if isinstance(X_participant, pd.DataFrame):
                X_participant = X_participant.values
            if isinstance(y_participant, pd.Series):
                y_participant = y_participant.values
                
            return {
                'participant_id': participant_id,
                'features': X_participant.tolist(),
                'label': y_participant.tolist()
            }
        except Exception as e:
            self.logger.error(f"創建參與者數據失敗: {str(e)}")
            return None
            
    def _add_noise_to_data(self, data: Dict[str, Any], noise_level: float = 0.1) -> Dict[str, Any]:
        """
        向數據添加噪聲以模擬真實場景
        
        Args:
            data: 原始數據
            noise_level: 噪聲水平
        """
        try:
            features = np.array(data['features'])
            noise = np.random.normal(0, noise_level, features.shape)
            noisy_features = features + noise
            noisy_features = np.clip(noisy_features, 0, 1)  # 確保值在 [0,1] 範圍內
            
            return {
                'participant_id': data['participant_id'],
                'features': noisy_features.tolist(),
                'label': data['label']
            }
        except Exception as e:
            self.logger.error(f"添加噪聲失敗: {str(e)}")
            return None
            
    def produce_data(self, num_participants: int = 10, interval: float = 1.0):
        """
        開始生產數據
        
        Args:
            num_participants: 參與者數量
            interval: 發送間隔（秒）
        """
        try:
            self.logger.info(f"開始生產數據，參與者數量: {num_participants}")
            while True:
                # 隨機選擇一個參與者
                participant_id = random.randint(1, num_participants)
                
                # 創建參與者數據
                data = self._create_participant_data(participant_id)
                if data is None:
                    continue
                
                # 添加噪聲
                noisy_data = self._add_noise_to_data(data)
                if noisy_data is None:
                    continue
                
                # 發送到 Kafka
                future = self.producer.send(self.topic, value=noisy_data)
                # 等待消息發送完成
                future.get(timeout=10)
                self.logger.info(f"成功發送數據到參與者 {participant_id}")
                
                # 等待指定間隔
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("停止數據生產")
        except Exception as e:
            self.logger.error(f"生產數據時發生錯誤: {str(e)}")
        finally:
            self.producer.close()
            
    def produce_batch(self, num_participants: int = 10, batch_size: int = 100):
        """
        生產一批數據
        
        Args:
            num_participants: 參與者數量
            batch_size: 批次大小
        """
        try:
            self.logger.info(f"開始生產批次數據，批次大小: {batch_size}")
            for i in range(batch_size):
                participant_id = random.randint(1, num_participants)
                data = self._create_participant_data(participant_id)
                if data is None:
                    continue
                    
                noisy_data = self._add_noise_to_data(data)
                if noisy_data is None:
                    continue
                
                future = self.producer.send(self.topic, value=noisy_data)
                future.get(timeout=10)
                self.logger.info(f"發送批次數據 {i+1}/{batch_size} 到參與者 {participant_id}")
                
            self.producer.flush()
            self.logger.info(f"完成發送 {batch_size} 條數據")
            
        except Exception as e:
            self.logger.error(f"生產批次數據時發生錯誤: {str(e)}")
            self.producer.close()

if __name__ == "__main__":
    # 創建生產者實例
    producer = MNISTDataProducer()
    
    # 開始生產數據
    producer.produce_data(num_participants=10, interval=0.1) 