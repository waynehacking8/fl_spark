#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DPConfig:
    """差分隱私配置"""
    epsilon: float = 1.0  # 隱私預算
    delta: float = 1e-5   # 鬆弛參數
    sensitivity: float = 1.0  # 敏感度
    clip_norm: float = 1.0    # 梯度裁剪範數

class DifferentialPrivacy:
    def __init__(self, config: DPConfig):
        """
        初始化差分隱私機制
        
        Args:
            config: 差分隱私配置
        """
        self.config = config
        self._privacy_spent = 0.0
        
    def add_noise(self, weights: np.ndarray) -> np.ndarray:
        """
        為權重添加高斯噪聲
        
        Args:
            weights: 模型權重
            
        Returns:
            添加噪聲後的權重
        """
        # 計算噪聲標準差
        sigma = np.sqrt(2 * np.log(1.25/self.config.delta)) * \
                self.config.sensitivity / self.config.epsilon
                
        # 生成高斯噪聲
        noise = np.random.normal(0, sigma, weights.shape)
        
        # 更新隱私開銷
        self._privacy_spent += self.config.epsilon
        
        return weights + noise
        
    def clip_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """
        裁剪梯度以限制敏感度
        
        Args:
            gradients: 原始梯度
            
        Returns:
            裁剪後的梯度
        """
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > self.config.clip_norm:
            gradients = gradients * self.config.clip_norm / grad_norm
        return gradients
        
    @property
    def privacy_spent(self) -> float:
        """獲取已消耗的隱私預算"""
        return self._privacy_spent
        
    def get_metrics(self) -> Dict[str, Any]:
        """獲取差分隱私相關指標"""
        return {
            'privacy_budget': self.config.epsilon,
            'privacy_spent': self.privacy_spent,
            'noise_scale': self.config.sensitivity / self.config.epsilon,
            'clip_norm': self.config.clip_norm
        } 