#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from typing import Dict, List, Any

class ExperimentEvaluator:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.results_dir = f"results/{experiment_name}"
        self.metrics = {
            'training_time': [],
            'communication_cost': [],
            'accuracy': [],
            'loss': [],
            'rounds': [],
            'timestamps': []
        }
        self.start_time = None
        self.communication_bytes = 0
        
        # 創建結果目錄
        os.makedirs(self.results_dir, exist_ok=True)
    
    def start_experiment(self):
        """開始實驗計時"""
        self.start_time = datetime.now()
    
    def record_metrics(self, round_num: int, accuracy: float, loss: float, 
                      communication_bytes: int = 0):
        """記錄實驗指標"""
        if self.start_time is None:
            raise ValueError("實驗尚未開始，請先調用 start_experiment()")
        
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        self.metrics['rounds'].append(round_num)
        self.metrics['training_time'].append(elapsed_time)
        self.metrics['accuracy'].append(accuracy)
        self.metrics['loss'].append(loss)
        self.metrics['communication_cost'].append(communication_bytes)
        self.metrics['timestamps'].append(datetime.now().isoformat())
        
        # 更新總通信成本
        self.communication_bytes += communication_bytes
    
    def save_results(self):
        """保存實驗結果"""
        # 保存指標數據
        df = pd.DataFrame(self.metrics)
        df.to_csv(f"{self.results_dir}/metrics.csv", index=False)
        
        # 保存實驗配置
        config = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_communication_bytes': self.communication_bytes,
            'total_training_time': self.metrics['training_time'][-1] if self.metrics['training_time'] else 0
        }
        
        with open(f"{self.results_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=4)
    
    def plot_results(self):
        """繪製實驗結果圖表"""
        # 設置圖表風格
        plt.style.use('seaborn')
        
        # 創建圖表目錄
        plots_dir = f"{self.results_dir}/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # 繪製準確率曲線
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['rounds'], self.metrics['accuracy'])
        plt.xlabel('訓練輪次')
        plt.ylabel('準確率 (%)')
        plt.title('訓練準確率隨輪次變化')
        plt.grid(True)
        plt.savefig(f"{plots_dir}/accuracy.png")
        plt.close()
        
        # 繪製損失曲線
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['rounds'], self.metrics['loss'])
        plt.xlabel('訓練輪次')
        plt.ylabel('損失')
        plt.title('訓練損失隨輪次變化')
        plt.grid(True)
        plt.savefig(f"{plots_dir}/loss.png")
        plt.close()
        
        # 繪製通信成本曲線
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['rounds'], self.metrics['communication_cost'])
        plt.xlabel('訓練輪次')
        plt.ylabel('通信成本 (bytes)')
        plt.title('每輪通信成本')
        plt.grid(True)
        plt.savefig(f"{plots_dir}/communication_cost.png")
        plt.close()
    
    def generate_report(self):
        """生成實驗報告"""
        report = f"""
# 聯邦學習實驗報告

## 實驗基本信息
- 實驗名稱：{self.experiment_name}
- 開始時間：{self.start_time.isoformat()}
- 結束時間：{datetime.now().isoformat()}
- 總訓練時間：{self.metrics['training_time'][-1]:.2f} 秒
- 總通信成本：{self.communication_bytes} bytes

## 性能指標
- 最終準確率：{self.metrics['accuracy'][-1]:.2f}%
- 最終損失：{self.metrics['loss'][-1]:.4f}
- 平均每輪訓練時間：{np.mean(self.metrics['training_time']):.2f} 秒
- 平均每輪通信成本：{np.mean(self.metrics['communication_cost']):.2f} bytes

## 訓練過程
- 總訓練輪次：{len(self.metrics['rounds'])}
- 準確率提升：{self.metrics['accuracy'][-1] - self.metrics['accuracy'][0]:.2f}%
- 損失降低：{self.metrics['loss'][0] - self.metrics['loss'][-1]:.4f}
        """
        
        with open(f"{self.results_dir}/report.md", 'w') as f:
            f.write(report)
    
    def compare_experiments(self, other_evaluator: 'ExperimentEvaluator'):
        """比較兩個實驗的結果"""
        comparison_dir = f"results/comparison_{self.experiment_name}_vs_{other_evaluator.experiment_name}"
        os.makedirs(comparison_dir, exist_ok=True)
        
        # 繪製準確率比較
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['rounds'], self.metrics['accuracy'], label=self.experiment_name)
        plt.plot(other_evaluator.metrics['rounds'], other_evaluator.metrics['accuracy'], 
                label=other_evaluator.experiment_name)
        plt.xlabel('訓練輪次')
        plt.ylabel('準確率 (%)')
        plt.title('準確率比較')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{comparison_dir}/accuracy_comparison.png")
        plt.close()
        
        # 繪製通信成本比較
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['rounds'], self.metrics['communication_cost'], 
                label=self.experiment_name)
        plt.plot(other_evaluator.metrics['rounds'], other_evaluator.metrics['communication_cost'], 
                label=other_evaluator.experiment_name)
        plt.xlabel('訓練輪次')
        plt.ylabel('通信成本 (bytes)')
        plt.title('通信成本比較')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{comparison_dir}/communication_cost_comparison.png")
        plt.close()
        
        # 生成比較報告
        comparison_report = f"""
# 聯邦學習實驗比較報告

## 實驗基本信息
- 實驗1：{self.experiment_name}
  - 總訓練時間：{self.metrics['training_time'][-1]:.2f} 秒
  - 總通信成本：{self.communication_bytes} bytes
  - 最終準確率：{self.metrics['accuracy'][-1]:.2f}%

- 實驗2：{other_evaluator.experiment_name}
  - 總訓練時間：{other_evaluator.metrics['training_time'][-1]:.2f} 秒
  - 總通信成本：{other_evaluator.communication_bytes} bytes
  - 最終準確率：{other_evaluator.metrics['accuracy'][-1]:.2f}%

## 性能比較
- 訓練時間差異：{self.metrics['training_time'][-1] - other_evaluator.metrics['training_time'][-1]:.2f} 秒
- 通信成本差異：{self.communication_bytes - other_evaluator.communication_bytes} bytes
- 準確率差異：{self.metrics['accuracy'][-1] - other_evaluator.metrics['accuracy'][-1]:.2f}%
        """
        
        with open(f"{comparison_dir}/comparison_report.md", 'w') as f:
            f.write(comparison_report)
    
    def save_metrics(self):
        """保存評估指標到 CSV 文件"""
        # 保存結果到 results.csv
        results_file = os.path.join(self.results_dir, 'results.csv')
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        # 如果文件不存在，創建並寫入標題
        if not os.path.exists(results_file):
            with open(results_file, 'w') as f:
                f.write('Round,Timestamp,Accuracy,Loss\n')
        
        # 追加新的結果
        with open(results_file, 'a') as f:
            for i in range(len(self.metrics['rounds'])):
                f.write(f"{self.metrics['rounds'][i]},{self.metrics['timestamps'][i]},{self.metrics['accuracy'][i]},{self.metrics['loss'][i]}\n")
        
        # 保存準確率歷史到 accuracy.csv
        accuracy_file = os.path.join(self.results_dir, 'spark_fl_accuracy.csv')
        os.makedirs(os.path.dirname(accuracy_file), exist_ok=True)
        
        # 如果文件不存在，創建並寫入標題
        if not os.path.exists(accuracy_file):
            with open(accuracy_file, 'w') as f:
                f.write('Round,Accuracy\n')
        
        # 追加新的準確率
        with open(accuracy_file, 'a') as f:
            for i in range(len(self.metrics['rounds'])):
                f.write(f"{self.metrics['rounds'][i]},{self.metrics['accuracy'][i]}\n")
        
        # 創建性能圖表
        self._plot_performance()
    
    def _plot_performance(self):
        """繪製性能圖表"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        color = 'tab:red'
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy (%)', color=color)
        ax1.plot(self.metrics['rounds'], self.metrics['accuracy'], color=color, marker='o', label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Loss', color=color)
        ax2.plot(self.metrics['rounds'], self.metrics['loss'], color=color, marker='x', linestyle='--', label='Loss')
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.suptitle('Spark FL Performance Over Rounds')
        fig.tight_layout()
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='center right')
        
        plot_file = os.path.join(self.results_dir, 'performance.png')
        plt.savefig(plot_file)
        plt.close(fig) 