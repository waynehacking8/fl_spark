#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp2 Worker Node Failure: Traditional FL vs Spark FL Performance Comparison
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

def create_exp2_comparison_plot(traditional_df, spark_df, save_path):
    """Create Exp2 specific comparison chart"""
    
    # Create 2x2 subplots with more spacing
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Exp2: Worker Node Failure - Traditional FL vs Spark FL Performance Comparison", 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Color configuration
    traditional_color = '#2E86AB'  # Blue
    spark_color = '#A23B72'       # Purple-red
    
    # 1. Accuracy Comparison with Worker Node Failure Analysis
    ax1.plot(traditional_df['Round'], traditional_df['Accuracy'], 
             color=traditional_color, linewidth=3, marker='o', markersize=6, 
             label='Traditional FL', alpha=0.8)
    ax1.plot(spark_df['Round'], spark_df['Accuracy'], 
             color=spark_color, linewidth=3, marker='s', markersize=6, 
             label='Spark FL', alpha=0.8)
    
    # Add worker node failure marker
    ax1.axvline(x=8, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(8.2, ax1.get_ylim()[1]*0.95, 'Worker Node Failure\n(Round 8)', 
            fontsize=10, color='red', fontweight='bold')
    
    # Highlight accuracy drop
    if len(traditional_df) > 7:
        ax1.annotate(f'Traditional FL: {traditional_df["Accuracy"].iloc[6]:.2f}% → {traditional_df["Accuracy"].iloc[7]:.2f}%',
                    xy=(8, traditional_df['Accuracy'].iloc[7]), xytext=(10, traditional_df['Accuracy'].iloc[7]-0.5),
                    arrowprops=dict(arrowstyle='->', color=traditional_color, alpha=0.7),
                    fontsize=9, color=traditional_color)
    
    if len(spark_df) > 7:
        ax1.annotate(f'Spark FL: {spark_df["Accuracy"].iloc[6]:.2f}% → {spark_df["Accuracy"].iloc[7]:.2f}%',
                    xy=(8, spark_df['Accuracy'].iloc[7]), xytext=(10, spark_df['Accuracy'].iloc[7]+0.3),
                    arrowprops=dict(arrowstyle='->', color=spark_color, alpha=0.7),
                    fontsize=9, color=spark_color)
    
    ax1.set_title('Accuracy Comparison with Fault Impact', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Recovery Time Analysis
    traditional_time_diff = traditional_df['Timestamp'].diff().fillna(traditional_df['Timestamp'].iloc[0])
    spark_time_diff = spark_df['Timestamp'].diff().fillna(spark_df['Timestamp'].iloc[0])
    
    ax2.bar(traditional_df['Round'] - 0.2, traditional_time_diff, 
            width=0.4, color=traditional_color, alpha=0.8, label='Traditional FL')
    ax2.bar(spark_df['Round'] + 0.2, spark_time_diff, 
            width=0.4, color=spark_color, alpha=0.8, label='Spark FL')
    
    # Highlight Round 8 recovery time
    ax2.axvline(x=8, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add recovery time annotations
    if len(traditional_time_diff) > 7:
        ax2.text(8, traditional_time_diff.iloc[7] + 2, 
                f'Traditional FL\nRecovery: {traditional_time_diff.iloc[7]:.1f}s', 
                ha='center', fontsize=9, color=traditional_color, fontweight='bold')
    
    if len(spark_time_diff) > 7:
        ax2.text(8, spark_time_diff.iloc[7] + 5, 
                f'Spark FL\nRecovery: {spark_time_diff.iloc[7]:.1f}s', 
                ha='center', fontsize=9, color=spark_color, fontweight='bold')
    
    ax2.set_title('Recovery Time Analysis (Time per Round)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Training Round', fontsize=12)
    ax2.set_ylabel('Time per Round (seconds)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Cumulative Time Comparison
    ax3.plot(traditional_df['Round'], traditional_df['Timestamp'], 
             color=traditional_color, linewidth=3, marker='o', markersize=6, 
             label='Traditional FL', alpha=0.8)
    ax3.plot(spark_df['Round'], spark_df['Timestamp'], 
             color=spark_color, linewidth=3, marker='s', markersize=6, 
             label='Spark FL', alpha=0.8)
    
    ax3.axvline(x=8, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax3.text(8.2, ax3.get_ylim()[1]*0.8, 'Worker Node Failure\n(Round 8)', 
            fontsize=10, color='red', fontweight='bold')
    
    ax3.set_title('Cumulative Training Time with Fault Impact', fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('Training Round', fontsize=12)
    ax3.set_ylabel('Cumulative Time (seconds)', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. Fault Tolerance Metrics
    # Calculate fault tolerance metrics
    traditional_final_acc = traditional_df['Accuracy'].iloc[-1]
    spark_final_acc = spark_df['Accuracy'].iloc[-1]
    traditional_final_time = traditional_df['Timestamp'].iloc[-1]
    spark_final_time = spark_df['Timestamp'].iloc[-1]
    
    # Recovery efficiency (accuracy maintained vs time cost)
    traditional_recovery_delay = traditional_df['Timestamp'].iloc[7] - traditional_df['Timestamp'].iloc[6] if len(traditional_df) > 7 else 0
    spark_recovery_delay = spark_df['Timestamp'].iloc[7] - spark_df['Timestamp'].iloc[6] if len(spark_df) > 7 else 0
    
    traditional_acc_drop = traditional_df['Accuracy'].iloc[6] - traditional_df['Accuracy'].iloc[7] if len(traditional_df) > 7 else 0
    spark_acc_drop = spark_df['Accuracy'].iloc[6] - spark_df['Accuracy'].iloc[7] if len(spark_df) > 7 else 0
    
    metrics = ['Final Accuracy\n(%)', 'Total Time\n(seconds)', 'Recovery Delay\n(seconds)', 'Accuracy Drop\n(%)']
    traditional_values = [traditional_final_acc, traditional_final_time, traditional_recovery_delay, traditional_acc_drop]
    spark_values = [spark_final_acc, spark_final_time, spark_recovery_delay, abs(spark_acc_drop)]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Normalize for visualization
    traditional_norm = [traditional_final_acc, traditional_final_time/10, traditional_recovery_delay, traditional_acc_drop*10]
    spark_norm = [spark_final_acc, spark_final_time/10, spark_recovery_delay, abs(spark_acc_drop)*10]
    
    bars1 = ax4.bar(x - width/2, traditional_norm, width, label='Traditional FL', 
                    color=traditional_color, alpha=0.8)
    bars2 = ax4.bar(x + width/2, spark_norm, width, label='Spark FL', 
                    color=spark_color, alpha=0.8)
    
    ax4.set_title('Fault Tolerance Performance Metrics', fontsize=14, fontweight='bold', pad=20)
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
                    f'{val1:.2f}%', ha='center', va='bottom', fontsize=9)
            ax4.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.5,
                    f'{val2:.2f}%', ha='center', va='bottom', fontsize=9)
        elif i == 1:  # Time
            ax4.text(bar1.get_x() + bar1.get_width()/2., height1 + 1,
                    f'{val1:.1f}s', ha='center', va='bottom', fontsize=9)
            ax4.text(bar2.get_x() + bar2.get_width()/2., height2 + 1,
                    f'{val2:.1f}s', ha='center', va='bottom', fontsize=9)
        elif i == 2:  # Recovery delay
            ax4.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.5,
                    f'{val1:.1f}s', ha='center', va='bottom', fontsize=9)
            ax4.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.5,
                    f'{val2:.1f}s', ha='center', va='bottom', fontsize=9)
        else:  # Accuracy drop
            ax4.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.2,
                    f'{val1:.2f}%', ha='center', va='bottom', fontsize=9)
            ax4.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.2,
                    f'{val2:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Exp2 comparison chart saved to: {save_path}")
    
    return fig

def main():
    """Main function"""
    
    # Create output directory
    output_dir = Path("comparison_plots")
    output_dir.mkdir(exist_ok=True)
    
    print("=== Exp2 Worker Node Failure: Traditional FL vs Spark FL Comparison ===")
    
    # Load Exp2 data
    traditional_path = "fault_tolerance/exp2_worker_node_failure/results/traditional/checkpoints/results.csv"
    spark_path = "fault_tolerance/exp2_worker_node_failure/spark_simple/results/spark_fl_results.csv"
    
    traditional_df, spark_df = load_data(traditional_path, spark_path)
    
    if traditional_df is not None and spark_df is not None:
        # Create specialized Exp2 comparison chart
        create_exp2_comparison_plot(
            traditional_df, spark_df,
            output_dir / "exp2_specialized_comparison.png"
        )
        
        # Print detailed statistics
        print(f"\n=== Exp2 Detailed Analysis ===")
        print(f"Traditional FL - Final accuracy: {traditional_df['Accuracy'].iloc[-1]:.2f}%, Final loss: {traditional_df['Loss'].iloc[-1]:.4f}, Total time: {traditional_df['Timestamp'].iloc[-1]:.1f}s")
        print(f"Spark FL - Final accuracy: {spark_df['Accuracy'].iloc[-1]:.2f}%, Final loss: {spark_df['Loss'].iloc[-1]:.4f}, Total time: {spark_df['Timestamp'].iloc[-1]:.1f}s")
        
        # Worker node failure impact analysis
        print(f"\n=== Worker Node Failure Impact (Round 8) ===")
        if len(traditional_df) > 7:
            trad_acc_before = traditional_df['Accuracy'].iloc[6]
            trad_acc_after = traditional_df['Accuracy'].iloc[7]
            trad_time_before = traditional_df['Timestamp'].iloc[6]
            trad_time_after = traditional_df['Timestamp'].iloc[7]
            trad_recovery_time = trad_time_after - trad_time_before
            trad_acc_drop = trad_acc_before - trad_acc_after
            
            print(f"Traditional FL:")
            print(f"  - Accuracy: {trad_acc_before:.2f}% → {trad_acc_after:.2f}% (drop: {trad_acc_drop:.2f}%)")
            print(f"  - Recovery time: {trad_recovery_time:.1f}s")
        
        if len(spark_df) > 7:
            spark_acc_before = spark_df['Accuracy'].iloc[6]
            spark_acc_after = spark_df['Accuracy'].iloc[7]
            spark_time_before = spark_df['Timestamp'].iloc[6]
            spark_time_after = spark_df['Timestamp'].iloc[7]
            spark_recovery_time = spark_time_after - spark_time_before
            spark_acc_drop = spark_acc_before - spark_acc_after
            
            print(f"Spark FL:")
            print(f"  - Accuracy: {spark_acc_before:.2f}% → {spark_acc_after:.2f}% (drop: {spark_acc_drop:.2f}%)")
            print(f"  - Recovery time: {spark_recovery_time:.1f}s")
        
        # Performance comparison
        time_advantage = ((traditional_df['Timestamp'].iloc[-1] - spark_df['Timestamp'].iloc[-1]) / traditional_df['Timestamp'].iloc[-1]) * 100
        print(f"\n=== Overall Performance ===")
        print(f"Spark FL time advantage: {time_advantage:.1f}% faster")
        print(f"Accuracy difference: {spark_df['Accuracy'].iloc[-1] - traditional_df['Accuracy'].iloc[-1]:.2f}%")
    
    print(f"\nExp2 specialized chart saved to {output_dir} directory")
    print("Visualization complete!")

if __name__ == "__main__":
    main() 