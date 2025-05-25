#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Traditional FL Worker Node Fault Tolerance Experiment Visualization
生成三個關鍵分析圖表：綜合分析、故障分析、時間線分析
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import os

# 設置中文字體和樣式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def load_data():
    """加載實驗數據"""
    results_file = "../results/traditional/checkpoints/results.csv"
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} rounds of data")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data range: Round {df['Round'].min()} to {df['Round'].max()}")
    
    return df

def create_comprehensive_analysis(df, output_dir):
    """創建4面板綜合分析圖"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Traditional FL Worker Node Fault Tolerance - Comprehensive Analysis', 
                 fontsize=16, fontweight='bold')
    
    rounds = df['Round']
    accuracy = df['Accuracy']
    loss = df['Loss']
    timestamps = df['Timestamp']
    
    # 計算每輪時間間隔
    round_durations = [timestamps.iloc[0]]  # 第一輪就是timestamp
    for i in range(1, len(timestamps)):
        duration = timestamps.iloc[i] - timestamps.iloc[i-1]
        round_durations.append(duration)
    
    # 面板1：準確率趨勢（標記故障點）
    ax1.plot(rounds, accuracy, 'b-o', linewidth=2, markersize=6, label='Test Accuracy')
    ax1.axvline(x=8, color='red', linestyle='--', alpha=0.7, label='Fault Injection (Round 8)')
    ax1.axvspan(7.5, 8.5, alpha=0.2, color='red', label='Fault Period')
    ax1.set_xlabel('Federated Round')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Model Accuracy with Fault Tolerance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([95, 100])
    
    # 標記關鍵點
    ax1.annotate(f'Pre-fault: {accuracy.iloc[6]:.2f}%', 
                xy=(7, accuracy.iloc[6]), xytext=(5, 97),
                arrowprops=dict(arrowstyle='->', color='green'))
    ax1.annotate(f'During fault: {accuracy.iloc[7]:.2f}%', 
                xy=(8, accuracy.iloc[7]), xytext=(10, 96),
                arrowprops=dict(arrowstyle='->', color='red'))
    ax1.annotate(f'Post-recovery: {accuracy.iloc[8]:.2f}%', 
                xy=(9, accuracy.iloc[8]), xytext=(12, 97),
                arrowprops=dict(arrowstyle='->', color='blue'))
    
    # 面板2：訓練損失趨勢
    ax2.plot(rounds, loss, 'g-s', linewidth=2, markersize=5, label='Training Loss')
    ax2.axvline(x=8, color='red', linestyle='--', alpha=0.7)
    ax2.axvspan(7.5, 8.5, alpha=0.2, color='red')
    ax2.set_xlabel('Federated Round')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Training Loss During Fault Events')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 面板3：每輪執行時間
    ax3.bar(rounds, round_durations, color=['red' if r == 8 else 'skyblue' for r in rounds], 
            alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Federated Round')
    ax3.set_ylabel('Round Duration (seconds)')
    ax3.set_title('Round Execution Time (Fault Detection Visible)')
    ax3.grid(True, alpha=0.3)
    
    # 標記第8輪特殊情況
    round8_duration = round_durations[7] if len(round_durations) > 7 else 0
    ax3.annotate(f'Fault Detection:\n{round8_duration:.1f}s (~30s)', 
                xy=(8, round8_duration), xytext=(12, round8_duration + 5),
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 面板4：故障恢復分析
    pre_fault_acc = accuracy.iloc[6]  # Round 7
    fault_acc = accuracy.iloc[7]      # Round 8  
    post_fault_acc = accuracy.iloc[8] if len(accuracy) > 8 else fault_acc  # Round 9
    final_acc = accuracy.iloc[-1]     # Final round
    
    categories = ['Pre-fault\n(Round 7)', 'During Fault\n(Round 8)', 
                  'Post-recovery\n(Round 9)', 'Final\n(Round 20)']
    values = [pre_fault_acc, fault_acc, post_fault_acc, final_acc]
    colors = ['green', 'red', 'blue', 'purple']
    
    bars = ax4.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax4.set_ylabel('Test Accuracy (%)')
    ax4.set_title('Fault Tolerance Recovery Analysis')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([98, 100])
    
    # 添加數值標籤
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.annotate(f'{value:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'traditional_fl_comprehensive_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Comprehensive analysis saved: {output_file}")
    return fig

def create_fault_analysis(df, output_dir):
    """創建故障分析雙軸圖"""
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    rounds = df['Round']
    accuracy = df['Accuracy']
    timestamps = df['Timestamp']
    
    # 計算參與者數量（第8輪只有2個參與者，其他都是4個）
    participants = [2 if r == 8 else 4 for r in rounds]
    
    # 主軸：準確率
    color1 = 'tab:blue'
    ax1.set_xlabel('Federated Round', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', color=color1, fontsize=12)
    line1 = ax1.plot(rounds, accuracy, 'o-', color=color1, linewidth=3, 
                     markersize=8, label='Test Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([98, 100])
    
    # 副軸：參與者數量
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Active Participants', color=color2, fontsize=12)
    bars = ax2.bar(rounds, participants, alpha=0.3, color=color2, 
                   edgecolor='red', linewidth=1, label='Active Participants')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([0, 5])
    
    # 標記故障期間
    ax1.axvspan(7.5, 8.5, alpha=0.2, color='red', label='Fault Period')
    ax1.axvline(x=8, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    # 標記關鍵事件
    pre_fault = accuracy.iloc[6]
    fault = accuracy.iloc[7] 
    post_fault = accuracy.iloc[8] if len(accuracy) > 8 else fault
    
    ax1.annotate('Participants 1&2 Fault\n30s Detection', 
                xy=(8, fault), xytext=(10, 98.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
                fontsize=11, fontweight='bold')
    
    ax1.annotate(f'Performance Drop:\n{pre_fault:.2f}% → {fault:.2f}%\n({fault-pre_fault:+.2f}%)', 
                xy=(8, fault-0.1), xytext=(5, 98.3),
                arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.6),
                fontsize=10)
    
    if len(accuracy) > 8:
        ax1.annotate(f'Quick Recovery:\n{post_fault:.2f}%', 
                    xy=(9, post_fault), xytext=(12, 99.2),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                    fontsize=11)
    
    # 圖例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    plt.title('Traditional FL Worker Node Fault Tolerance Analysis\n'
              'Dual-Axis: Model Performance vs Participant Availability', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'traditional_fl_fault_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Fault analysis saved: {output_file}")
    return fig

def create_timeline_analysis(df, output_dir):
    """創建時間線分析圖"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Traditional FL Worker Node Fault Tolerance - Timeline Analysis', 
                 fontsize=16, fontweight='bold')
    
    rounds = df['Round']
    accuracy = df['Accuracy'] 
    timestamps = df['Timestamp']
    
    # 計算每輪時間間隔
    round_durations = [timestamps.iloc[0]]
    for i in range(1, len(timestamps)):
        duration = timestamps.iloc[i] - timestamps.iloc[i-1] 
        round_durations.append(duration)
    
    # 上圖：累積時間vs準確率
    ax1.plot(timestamps, accuracy, 'b-o', linewidth=3, markersize=7, label='Test Accuracy')
    ax1.axvline(x=timestamps.iloc[7], color='red', linestyle='--', linewidth=2, alpha=0.8, 
                label='Fault Injection (Round 8)')
    ax1.axvspan(timestamps.iloc[6], timestamps.iloc[7], alpha=0.15, color='red', 
                label='Fault Detection Period')
    
    ax1.set_xlabel('Cumulative Time (seconds)')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Performance Timeline: Accuracy over Experiment Duration')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([98, 100])
    
    # 標記關鍵時間點
    fault_time = timestamps.iloc[7] if len(timestamps) > 7 else 0
    recovery_time = timestamps.iloc[8] if len(timestamps) > 8 else fault_time
    
    ax1.annotate(f'Fault Detection:\n{fault_time:.1f}s', 
                xy=(fault_time, accuracy.iloc[7]), xytext=(fault_time-20, 99.3),
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    if len(timestamps) > 8:
        ax1.annotate(f'Recovery Complete:\n{recovery_time:.1f}s', 
                    xy=(recovery_time, accuracy.iloc[8]), xytext=(recovery_time+15, 98.5),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # 下圖：每輪執行時間分析
    colors = []
    for i, r in enumerate(rounds):
        if r == 8:
            colors.append('red')
        elif r == 9:
            colors.append('orange') 
        else:
            colors.append('skyblue')
    
    bars = ax2.bar(rounds, round_durations, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Federated Round')
    ax2.set_ylabel('Round Duration (seconds)')
    ax2.set_title('Per-Round Execution Time Analysis')
    ax2.grid(True, alpha=0.3)
    
    # 標記關鍵輪次
    if len(round_durations) > 7:
        round8_duration = round_durations[7]
        ax2.annotate(f'Fault Detection:\n{round8_duration:.1f}s', 
                    xy=(8, round8_duration), xytext=(10, round8_duration + 3),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.2))
    
    if len(round_durations) > 8:
        round9_duration = round_durations[8]
        ax2.annotate(f'Quick Recovery:\n{round9_duration:.1f}s', 
                    xy=(9, round9_duration), xytext=(12, round9_duration + 2),
                    arrowprops=dict(arrowstyle='->', color='orange'),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.2))
    
    # 添加平均時間線
    normal_durations = [d for i, d in enumerate(round_durations) if rounds.iloc[i] not in [8, 9]]
    avg_normal = np.mean(normal_durations) if normal_durations else 0
    ax2.axhline(y=avg_normal, color='green', linestyle=':', linewidth=2, alpha=0.7,
                label=f'Normal Round Average: {avg_normal:.1f}s')
    ax2.legend()
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'traditional_fl_timeline_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Timeline analysis saved: {output_file}")
    return fig

def generate_summary_statistics(df):
    """生成實驗統計摘要"""
    rounds = df['Round']
    accuracy = df['Accuracy']
    timestamps = df['Timestamp']
    
    # 計算關鍵指標
    pre_fault_acc = accuracy.iloc[6] if len(accuracy) > 6 else 0  # Round 7
    fault_acc = accuracy.iloc[7] if len(accuracy) > 7 else 0      # Round 8
    post_fault_acc = accuracy.iloc[8] if len(accuracy) > 8 else 0 # Round 9
    final_acc = accuracy.iloc[-1]
    
    # 計算時間間隔
    fault_detection_time = timestamps.iloc[7] - timestamps.iloc[6] if len(timestamps) > 7 else 0
    recovery_time = timestamps.iloc[8] - timestamps.iloc[7] if len(timestamps) > 8 else 0
    total_time = timestamps.iloc[-1]
    
    # 性能影響
    performance_drop = fault_acc - pre_fault_acc
    recovery_improvement = post_fault_acc - fault_acc if post_fault_acc > 0 else 0
    
    summary = {
        "Pre-fault Accuracy (Round 7)": f"{pre_fault_acc:.2f}%",
        "During-fault Accuracy (Round 8)": f"{fault_acc:.2f}%", 
        "Post-recovery Accuracy (Round 9)": f"{post_fault_acc:.2f}%",
        "Final Accuracy": f"{final_acc:.2f}%",
        "Performance Drop": f"{performance_drop:+.2f}%",
        "Recovery Improvement": f"{recovery_improvement:+.2f}%",
        "Fault Detection Time": f"{fault_detection_time:.1f}s",
        "Recovery Time": f"{recovery_time:.1f}s", 
        "Total Experiment Time": f"{total_time:.1f}s",
        "Fault Tolerance Success": "✅ Yes" if abs(performance_drop) < 1.0 else "❌ No"
    }
    
    return summary

def main():
    """主函數"""
    print("🎯 Traditional FL Worker Node Fault Tolerance Visualization")
    print("=" * 60)
    
    # 創建輸出目錄
    output_dir = "../results/traditional/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加載數據
        df = load_data()
        
        # 生成統計摘要
        summary = generate_summary_statistics(df)
        print("\n📊 Experiment Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print(f"\n🎨 Generating visualizations...")
        
        # 生成圖表
        fig1 = create_comprehensive_analysis(df, output_dir)
        plt.close(fig1)
        
        fig2 = create_fault_analysis(df, output_dir) 
        plt.close(fig2)
        
        fig3 = create_timeline_analysis(df, output_dir)
        plt.close(fig3)
        
        print(f"\n🎉 All visualizations generated successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📈 Files generated:")
        print(f"  - traditional_fl_comprehensive_analysis.png")
        print(f"  - traditional_fl_fault_analysis.png") 
        print(f"  - traditional_fl_timeline_analysis.png")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 