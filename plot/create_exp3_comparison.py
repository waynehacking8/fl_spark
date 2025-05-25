#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp3 CIFAR-10: Traditional FL vs Spark FL Performance Comparison
Includes Normal and Exp1 (Data Shard Failure) modes
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set English font and style
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_data(traditional_path, spark_path):
    """Load Traditional FL and Spark FL data"""
    try:
        traditional_df = pd.read_csv(traditional_path)
        spark_df = pd.read_csv(spark_path)
        
        # Data cleaning
        if 'Loss' in traditional_df.columns:
            traditional_df['Loss'] = traditional_df['Loss'].astype(str).str.replace('l', '1')
            traditional_df['Loss'] = pd.to_numeric(traditional_df['Loss'], errors='coerce')
        
        if 'Loss' in spark_df.columns:
            spark_df['Loss'] = spark_df['Loss'].astype(str).str.replace('l', '1')
            spark_df['Loss'] = pd.to_numeric(spark_df['Loss'], errors='coerce')
        
        # Ensure numeric columns are numeric
        numeric_columns = ['Round', 'Timestamp', 'Accuracy', 'Loss']
        for col in numeric_columns:
            if col in traditional_df.columns:
                traditional_df[col] = pd.to_numeric(traditional_df[col], errors='coerce')
            if col in spark_df.columns:
                spark_df[col] = pd.to_numeric(spark_df[col], errors='coerce')
        
        # Remove rows with NaN
        traditional_df = traditional_df.dropna()
        spark_df = spark_df.dropna()
        
        print(f"Traditional FL data: {len(traditional_df)} rows")
        print(f"Spark FL data: {len(spark_df)} rows")
        
        return traditional_df, spark_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def create_exp3_comparison_plot(traditional_df, spark_df, title, save_path, experiment_type="normal"):
    """Create Exp3 CIFAR-10 specific comparison chart"""
    
    # Create 2x2 subplots with more spacing
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    # Color configuration
    traditional_color = '#2E86AB'  # Blue
    spark_color = '#A23B72'       # Purple-red
    
    # 1. Accuracy Comparison with CIFAR-10 Performance Analysis
    ax1.plot(traditional_df['Round'], traditional_df['Accuracy'], 
             color=traditional_color, linewidth=3, marker='o', markersize=6, 
             label='Traditional FL', alpha=0.8)
    ax1.plot(spark_df['Round'], spark_df['Accuracy'], 
             color=spark_color, linewidth=3, marker='s', markersize=6, 
             label='Spark FL', alpha=0.8)
    
    # Add fault marker for exp1
    if experiment_type == "exp1":
        ax1.axvline(x=5, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax1.text(5.2, ax1.get_ylim()[1]*0.95, 'Data Shard Failure\n(Round 5)', 
                fontsize=10, color='red', fontweight='bold')
        
        # Highlight accuracy impact for exp1
        if len(traditional_df) > 5:
            ax1.annotate(f'Traditional FL: {traditional_df["Accuracy"].iloc[4]:.1f}% → {traditional_df["Accuracy"].iloc[5]:.1f}%',
                        xy=(6, traditional_df['Accuracy'].iloc[5]), xytext=(8, traditional_df['Accuracy'].iloc[5]+2),
                        arrowprops=dict(arrowstyle='->', color=traditional_color, alpha=0.7),
                        fontsize=9, color=traditional_color)
        
        if len(spark_df) > 5:
            ax1.annotate(f'Spark FL: {spark_df["Accuracy"].iloc[4]:.1f}% → {spark_df["Accuracy"].iloc[5]:.1f}%',
                        xy=(6, spark_df['Accuracy'].iloc[5]), xytext=(8, spark_df['Accuracy'].iloc[5]-2),
                        arrowprops=dict(arrowstyle='->', color=spark_color, alpha=0.7),
                        fontsize=9, color=spark_color)
    
    ax1.set_title('CIFAR-10 Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss Comparison with Convergence Analysis
    ax2.plot(traditional_df['Round'], traditional_df['Loss'], 
             color=traditional_color, linewidth=3, marker='o', markersize=6, 
             label='Traditional FL', alpha=0.8)
    ax2.plot(spark_df['Round'], spark_df['Loss'], 
             color=spark_color, linewidth=3, marker='s', markersize=6, 
             label='Spark FL', alpha=0.8)
    
    if experiment_type == "exp1":
        ax2.axvline(x=5, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax2.text(5.2, ax2.get_ylim()[1]*0.9, 'Data Shard Failure\n(Round 5)', 
                fontsize=10, color='red', fontweight='bold')
    
    ax2.set_title('CIFAR-10 Loss Convergence', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Training Round', fontsize=12)
    ax2.set_ylabel('Loss Value', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Training Time Efficiency
    ax3.plot(traditional_df['Round'], traditional_df['Timestamp'], 
             color=traditional_color, linewidth=3, marker='o', markersize=6, 
             label='Traditional FL', alpha=0.8)
    ax3.plot(spark_df['Round'], spark_df['Timestamp'], 
             color=spark_color, linewidth=3, marker='s', markersize=6, 
             label='Spark FL', alpha=0.8)
    
    if experiment_type == "exp1":
        ax3.axvline(x=5, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax3.text(5.2, ax3.get_ylim()[1]*0.8, 'Data Shard Failure\n(Round 5)', 
                fontsize=10, color='red', fontweight='bold')
    
    ax3.set_title('CIFAR-10 Training Time Efficiency', fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('Training Round', fontsize=12)
    ax3.set_ylabel('Cumulative Time (seconds)', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. CIFAR-10 Performance Metrics
    traditional_final_acc = traditional_df['Accuracy'].iloc[-1]
    spark_final_acc = spark_df['Accuracy'].iloc[-1]
    traditional_final_loss = traditional_df['Loss'].iloc[-1]
    spark_final_loss = spark_df['Loss'].iloc[-1]
    traditional_final_time = traditional_df['Timestamp'].iloc[-1]
    spark_final_time = spark_df['Timestamp'].iloc[-1]
    
    # Calculate convergence speed (rounds to reach 85% accuracy)
    traditional_conv_round = len(traditional_df[traditional_df['Accuracy'] < 85]) + 1 if len(traditional_df[traditional_df['Accuracy'] < 85]) < len(traditional_df) else len(traditional_df)
    spark_conv_round = len(spark_df[spark_df['Accuracy'] < 85]) + 1 if len(spark_df[spark_df['Accuracy'] < 85]) < len(spark_df) else len(spark_df)
    
    metrics = ['Final Accuracy\n(%)', 'Final Loss', 'Total Time\n(seconds)', 'Convergence Speed\n(rounds to 85%)']
    traditional_values = [traditional_final_acc, traditional_final_loss, traditional_final_time, traditional_conv_round]
    spark_values = [spark_final_acc, spark_final_loss, spark_final_time, spark_conv_round]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Normalize for visualization
    traditional_norm = [traditional_final_acc, traditional_final_loss*100, traditional_final_time/100, traditional_conv_round*5]
    spark_norm = [spark_final_acc, spark_final_loss*100, spark_final_time/100, spark_conv_round*5]
    
    bars1 = ax4.bar(x - width/2, traditional_norm, width, label='Traditional FL', 
                    color=traditional_color, alpha=0.8)
    bars2 = ax4.bar(x + width/2, spark_norm, width, label='Spark FL', 
                    color=spark_color, alpha=0.8)
    
    ax4.set_title('CIFAR-10 Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    ax4.set_ylabel('Normalized Values', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, fontsize=10)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar1, bar2, val1, val2) in enumerate(zip(bars1, bars2, traditional_values, spark_values)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        if i == 0:  # Accuracy
            ax4.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.5,
                    f'{val1:.1f}%', ha='center', va='bottom', fontsize=9)
            ax4.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.5,
                    f'{val2:.1f}%', ha='center', va='bottom', fontsize=9)
        elif i == 1:  # Loss
            ax4.text(bar1.get_x() + bar1.get_width()/2., height1 + 2,
                    f'{val1:.3f}', ha='center', va='bottom', fontsize=9)
            ax4.text(bar2.get_x() + bar2.get_width()/2., height2 + 2,
                    f'{val2:.3f}', ha='center', va='bottom', fontsize=9)
        elif i == 2:  # Time
            ax4.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.5,
                    f'{val1:.0f}s', ha='center', va='bottom', fontsize=9)
            ax4.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.5,
                    f'{val2:.0f}s', ha='center', va='bottom', fontsize=9)
        else:  # Convergence
            ax4.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.5,
                    f'{val1:.0f}', ha='center', va='bottom', fontsize=9)
            ax4.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.5,
                    f'{val2:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Exp3 comparison chart saved to: {save_path}")
    
    return fig

def create_exp3_detailed_analysis(traditional_df, spark_df, title, save_path, experiment_type="normal"):
    """Create detailed analysis for Exp3 CIFAR-10"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'{title} - Detailed Analysis', fontsize=18, fontweight='bold', y=0.98)
    
    # Color configuration
    traditional_color = '#2E86AB'
    spark_color = '#A23B72'
    
    # 1. Learning Rate Analysis (Accuracy improvement per round)
    traditional_acc_diff = traditional_df['Accuracy'].diff().fillna(0)
    spark_acc_diff = spark_df['Accuracy'].diff().fillna(0)
    
    ax1.plot(traditional_df['Round'], traditional_acc_diff, 
             color=traditional_color, linewidth=2, marker='o', markersize=4, 
             label='Traditional FL', alpha=0.8)
    ax1.plot(spark_df['Round'], spark_acc_diff, 
             color=spark_color, linewidth=2, marker='s', markersize=4, 
             label='Spark FL', alpha=0.8)
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    if experiment_type == "exp1":
        ax1.axvline(x=5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    ax1.set_title('CIFAR-10 Learning Progress (Accuracy Change)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('Accuracy Change (%)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss Reduction Analysis
    traditional_loss_diff = traditional_df['Loss'].diff().fillna(0)
    spark_loss_diff = spark_df['Loss'].diff().fillna(0)
    
    ax2.plot(traditional_df['Round'], traditional_loss_diff, 
             color=traditional_color, linewidth=2, marker='o', markersize=4, 
             label='Traditional FL', alpha=0.8)
    ax2.plot(spark_df['Round'], spark_loss_diff, 
             color=spark_color, linewidth=2, marker='s', markersize=4, 
             label='Spark FL', alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    if experiment_type == "exp1":
        ax2.axvline(x=5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    ax2.set_title('CIFAR-10 Loss Reduction Trend', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Training Round', fontsize=12)
    ax2.set_ylabel('Loss Change', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Training Efficiency (Time per round)
    traditional_time_diff = traditional_df['Timestamp'].diff().fillna(traditional_df['Timestamp'].iloc[0])
    spark_time_diff = spark_df['Timestamp'].diff().fillna(spark_df['Timestamp'].iloc[0])
    
    ax3.bar(traditional_df['Round'] - 0.2, traditional_time_diff, 
            width=0.4, color=traditional_color, alpha=0.8, label='Traditional FL')
    ax3.bar(spark_df['Round'] + 0.2, spark_time_diff, 
            width=0.4, color=spark_color, alpha=0.8, label='Spark FL')
    
    if experiment_type == "exp1":
        ax3.axvline(x=5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    ax3.set_title('CIFAR-10 Training Efficiency (Time per Round)', fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('Training Round', fontsize=12)
    ax3.set_ylabel('Time per Round (seconds)', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Performance Stability Analysis (Moving averages)
    window = 3
    traditional_ma = traditional_df['Accuracy'].rolling(window=window).mean()
    spark_ma = spark_df['Accuracy'].rolling(window=window).mean()
    
    ax4.plot(traditional_df['Round'], traditional_df['Accuracy'], 
             color=traditional_color, alpha=0.3, linewidth=1, label='Traditional FL (Raw)')
    ax4.plot(traditional_df['Round'], traditional_ma, 
             color=traditional_color, linewidth=3, label='Traditional FL (3-Round Avg)')
    
    ax4.plot(spark_df['Round'], spark_df['Accuracy'], 
             color=spark_color, alpha=0.3, linewidth=1, label='Spark FL (Raw)')
    ax4.plot(spark_df['Round'], spark_ma, 
             color=spark_color, linewidth=3, label='Spark FL (3-Round Avg)')
    
    if experiment_type == "exp1":
        ax4.axvline(x=5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    ax4.set_title('CIFAR-10 Performance Stability Analysis', fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlabel('Training Round', fontsize=12)
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Exp3 detailed analysis chart saved to: {save_path}")
    
    return fig

def main():
    """Main function"""
    
    # Create output directory
    output_dir = Path("comparison_plots")
    output_dir.mkdir(exist_ok=True)
    
    print("=== Exp3 CIFAR-10: Traditional FL vs Spark FL Comparison ===")
    
    # 1. Normal mode comparison
    print("\n1. Processing Exp3 Normal mode data...")
    traditional_normal_path = "fault_tolerance/exp3_cifar10/results/traditional/normal/cifar10_normal_results.csv"
    spark_normal_path = "fault_tolerance/exp3_cifar10/results/spark/normal/cifar10_spark_normal_results.csv"
    
    traditional_df, spark_df = load_data(traditional_normal_path, spark_normal_path)
    
    if traditional_df is not None and spark_df is not None:
        # Create normal mode comparison chart
        create_exp3_comparison_plot(
            traditional_df, spark_df,
            "Exp3 CIFAR-10 Normal Mode: Traditional FL vs Spark FL Performance Comparison",
            output_dir / "exp3_normal_comparison.png",
            "normal"
        )
        
        # Create detailed analysis
        create_exp3_detailed_analysis(
            traditional_df, spark_df,
            "Exp3 CIFAR-10 Normal Mode: Traditional FL vs Spark FL",
            output_dir / "exp3_normal_detailed_analysis.png",
            "normal"
        )
        
        # Print statistics
        print(f"\nExp3 Normal mode statistics:")
        print(f"Traditional FL - Final accuracy: {traditional_df['Accuracy'].iloc[-1]:.2f}%, Final loss: {traditional_df['Loss'].iloc[-1]:.4f}, Total time: {traditional_df['Timestamp'].iloc[-1]:.1f}s")
        print(f"Spark FL - Final accuracy: {spark_df['Accuracy'].iloc[-1]:.2f}%, Final loss: {spark_df['Loss'].iloc[-1]:.4f}, Total time: {spark_df['Timestamp'].iloc[-1]:.1f}s")
        
        # Performance comparison
        time_advantage = ((traditional_df['Timestamp'].iloc[-1] - spark_df['Timestamp'].iloc[-1]) / traditional_df['Timestamp'].iloc[-1]) * 100
        print(f"Spark FL time advantage: {time_advantage:.1f}% faster")
        print(f"Accuracy difference: {spark_df['Accuracy'].iloc[-1] - traditional_df['Accuracy'].iloc[-1]:.2f}%")
    
    # 2. Exp1 mode comparison
    print("\n2. Processing Exp3 Exp1 mode data...")
    traditional_exp1_path = "fault_tolerance/exp3_cifar10/results/traditional/exp1/cifar10_exp1_results.csv"
    spark_exp1_path = "fault_tolerance/exp3_cifar10/results/spark/exp1/cifar10_spark_exp1_results.csv"
    
    traditional_df, spark_df = load_data(traditional_exp1_path, spark_exp1_path)
    
    if traditional_df is not None and spark_df is not None:
        # Create exp1 mode comparison chart
        create_exp3_comparison_plot(
            traditional_df, spark_df,
            "Exp3 CIFAR-10 Exp1 Mode: Traditional FL vs Spark FL Fault Tolerance Comparison",
            output_dir / "exp3_exp1_comparison.png",
            "exp1"
        )
        
        # Create detailed analysis
        create_exp3_detailed_analysis(
            traditional_df, spark_df,
            "Exp3 CIFAR-10 Exp1 Mode: Traditional FL vs Spark FL",
            output_dir / "exp3_exp1_detailed_analysis.png",
            "exp1"
        )
        
        # Print statistics
        print(f"\nExp3 Exp1 mode statistics:")
        print(f"Traditional FL - Final accuracy: {traditional_df['Accuracy'].iloc[-1]:.2f}%, Final loss: {traditional_df['Loss'].iloc[-1]:.4f}, Total time: {traditional_df['Timestamp'].iloc[-1]:.1f}s")
        print(f"Spark FL - Final accuracy: {spark_df['Accuracy'].iloc[-1]:.2f}%, Final loss: {spark_df['Loss'].iloc[-1]:.4f}, Total time: {spark_df['Timestamp'].iloc[-1]:.1f}s")
        
        # Fault tolerance analysis
        print(f"\nData shard failure analysis (Round 5):")
        if len(traditional_df) > 5:
            print(f"Traditional FL - Round 4 accuracy: {traditional_df['Accuracy'].iloc[3]:.2f}%, Round 5 accuracy: {traditional_df['Accuracy'].iloc[4]:.2f}%, Round 6 accuracy: {traditional_df['Accuracy'].iloc[5]:.2f}%")
        if len(spark_df) > 5:
            print(f"Spark FL - Round 4 accuracy: {spark_df['Accuracy'].iloc[3]:.2f}%, Round 5 accuracy: {spark_df['Accuracy'].iloc[4]:.2f}%, Round 6 accuracy: {spark_df['Accuracy'].iloc[5]:.2f}%")
        
        # Performance comparison
        time_advantage = ((traditional_df['Timestamp'].iloc[-1] - spark_df['Timestamp'].iloc[-1]) / traditional_df['Timestamp'].iloc[-1]) * 100
        print(f"Spark FL time advantage: {time_advantage:.1f}% faster")
        print(f"Accuracy difference: {spark_df['Accuracy'].iloc[-1] - traditional_df['Accuracy'].iloc[-1]:.2f}%")
    
    print(f"\nAll Exp3 CIFAR-10 charts saved to {output_dir} directory")
    print("Visualization complete!")

if __name__ == "__main__":
    main() 