#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_performance():
    """根據 results.csv 生成性能圖表 (傳統FL風格)"""
    
    # 讀取結果文件
    results_path = '../results/traditional/checkpoints/results.csv'
    if not os.path.exists(results_path):
        print(f"❌ 找不到結果文件: {results_path}")
        return
    
    df = pd.read_csv(results_path)
    print(f"✅ 讀取到 {len(df)} 輪實驗數據")
    
    # 設置圖表風格 (完全按照傳統FL風格)
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    x_values = range(1, len(df) + 1)
    
    # 準確率曲線 (傳統FL標準風格)
    color = 'tab:red'
    ax1.plot(x_values, df['Accuracy'], color=color, marker='o', label='Accuracy')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    ax1.set_ylim(80, 100)  # 標準80-100%範圍
    ax1.legend(loc='lower right')
    
    # 標注第8輪故障點 (簡潔風格)
    if len(df) >= 8:
        ax1.axvline(x=8, color='gray', linestyle='--', alpha=0.7)
        ax1.text(8.5, 82, 'Round 8\nFault Tolerance\n(60s timeout)', 
                fontsize=9, color='gray', ha='left')
    
    # 損失曲線 (傳統FL標準風格)
    color = 'tab:blue'
    ax2.plot(x_values, df['Loss'], color=color, marker='x', label='Loss')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Loss', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(True)
    ax2.legend(loc='upper right')
    
    # 調整子圖間距
    plt.tight_layout()
    
    # 保存圖表 (標準參數)
    output_path = '../results/traditional/checkpoints/performance.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    print(f"🎨 傳統FL風格圖表已生成: {output_path}")
    
    # 打印關鍵統計
    print(f"\n📊 實驗統計:")
    print(f"   🔢 總輪次: {len(df)}")
    print(f"   🎯 最終準確率: {df['Accuracy'].iloc[-1]:.2f}%")
    print(f"   📉 最終損失: {df['Loss'].iloc[-1]:.4f}")
    print(f"   ⚡ 第8輪容錯延遲: {df['Timestamp'].iloc[7] - df['Timestamp'].iloc[6]:.1f}秒")
    print(f"   ✅ 自動編號修正: 連續輪次 1-{len(df)}")

if __name__ == "__main__":
    print("🎨 生成聯邦學習性能圖表...")
    plot_performance() 