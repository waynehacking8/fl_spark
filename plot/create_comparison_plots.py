#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Traditional FL vs Spark FL Performance Comparison Visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.patches as mpatches

# Set English font and style
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_data(traditional_path, spark_path):
    """Load Traditional FL and Spark FL data"""
    try:
        traditional_df = pd.read_csv(traditional_path)
        spark_df = pd.read_csv(spark_path)
        
        # Data cleaning: handle possible format errors
        # Fix Loss column character errors (e.g., '0.0l' -> '0.01')
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

def create_comparison_plot(traditional_df, spark_df, title, save_path, experiment_type="normal"):
    """Create comparison charts"""
    
    # Create 2x2 subplots with more spacing
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
    
    # Color configuration
    traditional_color = '#2E86AB'  # Blue
    spark_color = '#A23B72'       # Purple-red
    
    # 1. Accuracy Comparison (top left)
    ax1.plot(traditional_df['Round'], traditional_df['Accuracy'], 
             color=traditional_color, linewidth=3, marker='o', markersize=6, 
             label='Traditional FL', alpha=0.8)
    ax1.plot(spark_df['Round'], spark_df['Accuracy'], 
             color=spark_color, linewidth=3, marker='s', markersize=6, 
             label='Spark FL', alpha=0.8)
    
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add fault marker (if exp1)
    if experiment_type == "exp1":
        ax1.axvline(x=5, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax1.text(5.2, ax1.get_ylim()[1]*0.95, 'Fault Injection\n(Round 5)', 
                fontsize=10, color='red', fontweight='bold')
    elif experiment_type == "exp2":
        ax1.axvline(x=8, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax1.text(8.2, ax1.get_ylim()[1]*0.95, 'Worker Node Failure\n(Round 8)', 
                fontsize=10, color='red', fontweight='bold')
    
    # 2. Loss Comparison (top right)
    ax2.plot(traditional_df['Round'], traditional_df['Loss'], 
             color=traditional_color, linewidth=3, marker='o', markersize=6, 
             label='Traditional FL', alpha=0.8)
    ax2.plot(spark_df['Round'], spark_df['Loss'], 
             color=spark_color, linewidth=3, marker='s', markersize=6, 
             label='Spark FL', alpha=0.8)
    
    ax2.set_title('Loss Comparison', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Training Round', fontsize=12)
    ax2.set_ylabel('Loss Value', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    if experiment_type == "exp1":
        ax2.axvline(x=5, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax2.text(5.2, ax2.get_ylim()[1]*0.9, 'Fault Injection\n(Round 5)', 
                fontsize=10, color='red', fontweight='bold')
    elif experiment_type == "exp2":
        ax2.axvline(x=8, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax2.text(8.2, ax2.get_ylim()[1]*0.9, 'Worker Node Failure\n(Round 8)', 
                fontsize=10, color='red', fontweight='bold')
    
    # 3. Training Time Comparison (bottom left)
    ax3.plot(traditional_df['Round'], traditional_df['Timestamp'], 
             color=traditional_color, linewidth=3, marker='o', markersize=6, 
             label='Traditional FL', alpha=0.8)
    ax3.plot(spark_df['Round'], spark_df['Timestamp'], 
             color=spark_color, linewidth=3, marker='s', markersize=6, 
             label='Spark FL', alpha=0.8)
    
    ax3.set_title('Cumulative Training Time Comparison', fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('Training Round', fontsize=12)
    ax3.set_ylabel('Cumulative Time (seconds)', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    if experiment_type == "exp1":
        ax3.axvline(x=5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    elif experiment_type == "exp2":
        ax3.axvline(x=8, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # 4. Performance Statistics Comparison (bottom right)
    # Calculate statistics
    traditional_final_acc = traditional_df['Accuracy'].iloc[-1]
    spark_final_acc = spark_df['Accuracy'].iloc[-1]
    traditional_final_loss = traditional_df['Loss'].iloc[-1]
    spark_final_loss = spark_df['Loss'].iloc[-1]
    traditional_time = traditional_df['Timestamp'].iloc[-1]
    spark_time = spark_df['Timestamp'].iloc[-1]
    
    # Create bar chart
    metrics = ['Final Accuracy\n(%)', 'Final Loss', 'Total Training Time\n(seconds)']
    traditional_values = [traditional_final_acc, traditional_final_loss, traditional_time]
    spark_values = [spark_final_acc, spark_final_loss, spark_time]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Normalize data for comparison
    traditional_norm = [traditional_final_acc, traditional_final_loss*100, traditional_time/10]
    spark_norm = [spark_final_acc, spark_final_loss*100, spark_time/10]
    
    bars1 = ax4.bar(x - width/2, traditional_norm, width, label='Traditional FL', 
                    color=traditional_color, alpha=0.8)
    bars2 = ax4.bar(x + width/2, spark_norm, width, label='Spark FL', 
                    color=spark_color, alpha=0.8)
    
    ax4.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
    ax4.set_ylabel('Normalized Values', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, fontsize=10)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        if i == 0:  # Accuracy
            ax4.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.5,
                    f'{traditional_final_acc:.2f}%', ha='center', va='bottom', fontsize=9)
            ax4.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.5,
                    f'{spark_final_acc:.2f}%', ha='center', va='bottom', fontsize=9)
        elif i == 1:  # Loss
            ax4.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.5,
                    f'{traditional_final_loss:.4f}', ha='center', va='bottom', fontsize=9)
            ax4.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.5,
                    f'{spark_final_loss:.4f}', ha='center', va='bottom', fontsize=9)
        else:  # Time
            ax4.text(bar1.get_x() + bar1.get_width()/2., height1 + 2,
                    f'{traditional_time:.1f}s', ha='center', va='bottom', fontsize=9)
            ax4.text(bar2.get_x() + bar2.get_width()/2., height2 + 2,
                    f'{spark_time:.1f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Chart saved to: {save_path}")
    
    return fig

def create_detailed_analysis(traditional_df, spark_df, title, save_path, experiment_type="normal"):
    """Create detailed analysis charts"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'{title} - Detailed Analysis', fontsize=20, fontweight='bold', y=0.98)
    
    # Color configuration
    traditional_color = '#2E86AB'
    spark_color = '#A23B72'
    
    # 1. Accuracy Improvement Trend
    traditional_acc_diff = traditional_df['Accuracy'].diff().fillna(0)
    spark_acc_diff = spark_df['Accuracy'].diff().fillna(0)
    
    ax1.plot(traditional_df['Round'], traditional_acc_diff, 
             color=traditional_color, linewidth=2, marker='o', markersize=4, 
             label='Traditional FL', alpha=0.8)
    ax1.plot(spark_df['Round'], spark_acc_diff, 
             color=spark_color, linewidth=2, marker='s', markersize=4, 
             label='Spark FL', alpha=0.8)
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title('Accuracy Improvement Trend', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('Accuracy Change (%)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    if experiment_type == "exp1":
        ax1.axvline(x=5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    elif experiment_type == "exp2":
        ax1.axvline(x=8, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # 2. Loss Change Trend
    traditional_loss_diff = traditional_df['Loss'].diff().fillna(0)
    spark_loss_diff = spark_df['Loss'].diff().fillna(0)
    
    ax2.plot(traditional_df['Round'], traditional_loss_diff, 
             color=traditional_color, linewidth=2, marker='o', markersize=4, 
             label='Traditional FL', alpha=0.8)
    ax2.plot(spark_df['Round'], spark_loss_diff, 
             color=spark_color, linewidth=2, marker='s', markersize=4, 
             label='Spark FL', alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('Loss Change Trend', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Training Round', fontsize=12)
    ax2.set_ylabel('Loss Change', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    if experiment_type == "exp1":
        ax2.axvline(x=5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    elif experiment_type == "exp2":
        ax2.axvline(x=8, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # 3. Time per Round
    traditional_time_diff = traditional_df['Timestamp'].diff().fillna(traditional_df['Timestamp'].iloc[0])
    spark_time_diff = spark_df['Timestamp'].diff().fillna(spark_df['Timestamp'].iloc[0])
    
    ax3.bar(traditional_df['Round'] - 0.2, traditional_time_diff, 
            width=0.4, color=traditional_color, alpha=0.8, label='Traditional FL')
    ax3.bar(spark_df['Round'] + 0.2, spark_time_diff, 
            width=0.4, color=spark_color, alpha=0.8, label='Spark FL')
    
    ax3.set_title('Time per Round', fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('Training Round', fontsize=12)
    ax3.set_ylabel('Time (seconds)', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    if experiment_type == "exp1":
        ax3.axvline(x=5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    elif experiment_type == "exp2":
        ax3.axvline(x=8, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # 4. Convergence Analysis
    # Calculate moving average
    window = 3
    traditional_ma = traditional_df['Accuracy'].rolling(window=window).mean()
    spark_ma = spark_df['Accuracy'].rolling(window=window).mean()
    
    ax4.plot(traditional_df['Round'], traditional_df['Accuracy'], 
             color=traditional_color, alpha=0.3, linewidth=1, label='Traditional FL (Raw)')
    ax4.plot(traditional_df['Round'], traditional_ma, 
             color=traditional_color, linewidth=3, label='Traditional FL (Moving Avg)')
    
    ax4.plot(spark_df['Round'], spark_df['Accuracy'], 
             color=spark_color, alpha=0.3, linewidth=1, label='Spark FL (Raw)')
    ax4.plot(spark_df['Round'], spark_ma, 
             color=spark_color, linewidth=3, label='Spark FL (Moving Avg)')
    
    ax4.set_title('Convergence Analysis', fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlabel('Training Round', fontsize=12)
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    if experiment_type == "exp1":
        ax4.axvline(x=5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    elif experiment_type == "exp2":
        ax4.axvline(x=8, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Detailed analysis chart saved to: {save_path}")
    
    return fig

def main():
    """Main function"""
    
    # Create output directory
    output_dir = Path("comparison_plots")
    output_dir.mkdir(exist_ok=True)
    
    print("=== Traditional FL vs Spark FL Performance Comparison Visualization ===")
    
    # 1. Original experiment comparison
    print("\n1. Processing Original experiment data...")
    traditional_path = "original/results/traditional/checkpoints/results.csv"
    spark_path = "original/results/spark/results.csv"
    
    traditional_df, spark_df = load_data(traditional_path, spark_path)
    
    if traditional_df is not None and spark_df is not None:
        # Basic comparison chart
        create_comparison_plot(
            traditional_df, spark_df,
            "Original Experiment: Traditional FL vs Spark FL Performance Comparison",
            output_dir / "original_comparison.png",
            "normal"
        )
        
        # Detailed analysis chart
        create_detailed_analysis(
            traditional_df, spark_df,
            "Original Experiment: Traditional FL vs Spark FL",
            output_dir / "original_detailed_analysis.png",
            "normal"
        )
        
        # Print statistics
        print(f"\nOriginal experiment statistics:")
        print(f"Traditional FL - Final accuracy: {traditional_df['Accuracy'].iloc[-1]:.2f}%, Final loss: {traditional_df['Loss'].iloc[-1]:.4f}, Total time: {traditional_df['Timestamp'].iloc[-1]:.1f}s")
        print(f"Spark FL - Final accuracy: {spark_df['Accuracy'].iloc[-1]:.2f}%, Final loss: {spark_df['Loss'].iloc[-1]:.4f}, Total time: {spark_df['Timestamp'].iloc[-1]:.1f}s")
    
    # 2. Exp1 experiment comparison
    print("\n2. Processing Exp1 experiment data...")
    traditional_path = "fault_tolerance/exp1_data_shard_failure/results/traditional/checkpoints/results.csv"
    spark_path = "fault_tolerance/exp1_data_shard_failure/results/spark/results.csv"
    
    traditional_df, spark_df = load_data(traditional_path, spark_path)
    
    if traditional_df is not None and spark_df is not None:
        # Basic comparison chart
        create_comparison_plot(
            traditional_df, spark_df,
            "Exp1 Experiment: Traditional FL vs Spark FL Fault Tolerance Performance Comparison",
            output_dir / "exp1_comparison.png",
            "exp1"
        )
        
        # Detailed analysis chart
        create_detailed_analysis(
            traditional_df, spark_df,
            "Exp1 Experiment: Traditional FL vs Spark FL",
            output_dir / "exp1_detailed_analysis.png",
            "exp1"
        )
        
        # Print statistics
        print(f"\nExp1 experiment statistics:")
        print(f"Traditional FL - Final accuracy: {traditional_df['Accuracy'].iloc[-1]:.2f}%, Final loss: {traditional_df['Loss'].iloc[-1]:.4f}, Total time: {traditional_df['Timestamp'].iloc[-1]:.1f}s")
        print(f"Spark FL - Final accuracy: {spark_df['Accuracy'].iloc[-1]:.2f}%, Final loss: {spark_df['Loss'].iloc[-1]:.4f}, Total time: {spark_df['Timestamp'].iloc[-1]:.1f}s")
        
        # Fault recovery analysis
        print(f"\nFault recovery analysis (Round 5):")
        if len(traditional_df) > 5:
            print(f"Traditional FL - Round 5 accuracy: {traditional_df['Accuracy'].iloc[4]:.2f}%, Round 6 accuracy: {traditional_df['Accuracy'].iloc[5]:.2f}%")
        if len(spark_df) > 5:
            print(f"Spark FL - Round 5 accuracy: {spark_df['Accuracy'].iloc[4]:.2f}%, Round 6 accuracy: {spark_df['Accuracy'].iloc[5]:.2f}%")
    
    # 3. Exp2 experiment comparison
    print("\n3. Processing Exp2 experiment data...")
    traditional_path = "fault_tolerance/exp2_worker_node_failure/results/traditional/checkpoints/results.csv"
    spark_path = "fault_tolerance/exp2_worker_node_failure/spark_simple/results/spark_fl_results.csv"
    
    traditional_df, spark_df = load_data(traditional_path, spark_path)
    
    if traditional_df is not None and spark_df is not None:
        # Basic comparison chart
        create_comparison_plot(
            traditional_df, spark_df,
            "Exp2 Experiment: Traditional FL vs Spark FL Worker Node Failure Performance Comparison",
            output_dir / "exp2_comparison.png",
            "exp2"
        )
        
        # Detailed analysis chart
        create_detailed_analysis(
            traditional_df, spark_df,
            "Exp2 Experiment: Traditional FL vs Spark FL",
            output_dir / "exp2_detailed_analysis.png",
            "exp2"
        )
        
        # Print statistics
        print(f"\nExp2 experiment statistics:")
        print(f"Traditional FL - Final accuracy: {traditional_df['Accuracy'].iloc[-1]:.2f}%, Final loss: {traditional_df['Loss'].iloc[-1]:.4f}, Total time: {traditional_df['Timestamp'].iloc[-1]:.1f}s")
        print(f"Spark FL - Final accuracy: {spark_df['Accuracy'].iloc[-1]:.2f}%, Final loss: {spark_df['Loss'].iloc[-1]:.4f}, Total time: {spark_df['Timestamp'].iloc[-1]:.1f}s")
        
        # Worker node failure analysis
        print(f"\nWorker node failure analysis (Round 8):")
        if len(traditional_df) > 8:
            print(f"Traditional FL - Round 7 accuracy: {traditional_df['Accuracy'].iloc[6]:.2f}%, Round 8 accuracy: {traditional_df['Accuracy'].iloc[7]:.2f}%")
            print(f"Traditional FL - Round 7 time: {traditional_df['Timestamp'].iloc[6]:.1f}s, Round 8 time: {traditional_df['Timestamp'].iloc[7]:.1f}s (delay: {traditional_df['Timestamp'].iloc[7] - traditional_df['Timestamp'].iloc[6]:.1f}s)")
        if len(spark_df) > 8:
            print(f"Spark FL - Round 7 accuracy: {spark_df['Accuracy'].iloc[6]:.2f}%, Round 8 accuracy: {spark_df['Accuracy'].iloc[7]:.2f}%")
            print(f"Spark FL - Round 7 time: {spark_df['Timestamp'].iloc[6]:.1f}s, Round 8 time: {spark_df['Timestamp'].iloc[7]:.1f}s (delay: {spark_df['Timestamp'].iloc[7] - spark_df['Timestamp'].iloc[6]:.1f}s)")
    
    print(f"\nAll charts saved to {output_dir} directory")
    print("Visualization complete!")

if __name__ == "__main__":
    main() 