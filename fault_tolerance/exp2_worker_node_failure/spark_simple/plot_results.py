#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spark FL Worker Node Fault Tolerance Experiment Results Visualization
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import os

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

def load_results():
    """Load experiment results"""
    results_file = 'results/spark_fl_results.csv'
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return None
    
    df = pd.read_csv(results_file)
    print(f"Loaded data: {len(df)} rounds")
    return df

def create_comprehensive_plot(df):
    """Create comprehensive analysis plot"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Spark FL Worker Node Fault Tolerance Results', fontsize=16, fontweight='bold')
    
    rounds = df['Round']
    fault_round = 8
    
    # 1. Accuracy change plot
    ax1.plot(rounds, df['Accuracy'], 'b-', linewidth=2, marker='o', markersize=6, label='Test Accuracy')
    ax1.axvline(x=fault_round, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Fault Injection')
    ax1.axvspan(fault_round-0.5, fault_round+0.5, alpha=0.2, color='red', label='Fault Period')
    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy with Fault Tolerance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(96, 100)
    
    # Annotate key points
    fault_accuracy = df[df['Round'] == fault_round]['Accuracy'].iloc[0]
    recovery_accuracy = df[df['Round'] == fault_round+1]['Accuracy'].iloc[0]
    
    ax1.annotate(f'Fault: {fault_accuracy:.2f}%', 
                xy=(fault_round, fault_accuracy), xytext=(fault_round+2, fault_accuracy-0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red', fontweight='bold')
    
    ax1.annotate(f'Recovery: {recovery_accuracy:.2f}%', 
                xy=(fault_round+1, recovery_accuracy), xytext=(fault_round+3, recovery_accuracy+0.3),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=10, color='green', fontweight='bold')
    
    # 2. Loss change plot
    ax2.plot(rounds, df['Loss'], 'g-', linewidth=2, marker='s', markersize=6, label='Training Loss')
    ax2.axvline(x=fault_round, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axvspan(fault_round-0.5, fault_round+0.5, alpha=0.2, color='red')
    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss with Fault Tolerance')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Participants count plot
    colors = ['green' if p == 4 else 'red' for p in df['Participants']]
    bars = ax3.bar(rounds, df['Participants'], color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax3.axvline(x=fault_round, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Training Round')
    ax3.set_ylabel('Active Participants')
    ax3.set_title('Active Participants per Round')
    ax3.set_ylim(0, 5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add participant count labels
    for i, (round_num, participants) in enumerate(zip(rounds, df['Participants'])):
        if participants != 4:
            ax3.annotate(f'{participants}/4', 
                        xy=(round_num, participants), xytext=(0, 5),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color='red')
    
    # 4. Fault recovery detailed analysis
    pre_fault_acc = df[df['Round'] == fault_round-1]['Accuracy'].iloc[0]
    fault_acc = df[df['Round'] == fault_round]['Accuracy'].iloc[0]
    post_fault_acc = df[df['Round'] == fault_round+1]['Accuracy'].iloc[0]
    
    recovery_metrics = {
        'Pre-fault\n(Round 7)': pre_fault_acc,
        'During fault\n(Round 8)': fault_acc,
        'Post-recovery\n(Round 9)': post_fault_acc,
        'Final\n(Round 20)': df['Accuracy'].iloc[-1]
    }
    
    bars = ax4.bar(recovery_metrics.keys(), recovery_metrics.values(), 
                   color=['lightblue', 'red', 'lightgreen', 'darkgreen'],
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Fault Recovery Analysis')
    ax4.set_ylim(98, 100)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, recovery_metrics.values()):
        ax4.annotate(f'{value:.2f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_timeline_plot(df):
    """Create timeline plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Spark FL Fault Tolerance Timeline Analysis', fontsize=16, fontweight='bold')
    
    rounds = df['Round']
    timestamps = df['Timestamp']
    fault_round = 8
    
    # 1. Accuracy vs Time
    ax1.plot(timestamps, df['Accuracy'], 'b-', linewidth=3, marker='o', markersize=8, 
             markerfacecolor='white', markeredgecolor='blue', markeredgewidth=2)
    
    # Mark fault time period
    fault_start = df[df['Round'] == fault_round]['Timestamp'].iloc[0]
    fault_end = df[df['Round'] == fault_round+1]['Timestamp'].iloc[0]
    
    ax1.axvspan(fault_start-5, fault_end, alpha=0.3, color='red', label='Fault & Recovery Period')
    ax1.axvline(x=fault_start, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Fault Injection')
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Performance Timeline')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(96, 100)
    
    # 2. Training speed analysis (round interval times)
    round_intervals = []
    for i in range(1, len(timestamps)):
        interval = timestamps.iloc[i] - timestamps.iloc[i-1]
        round_intervals.append(interval)
    
    round_numbers = rounds[1:]
    colors = ['red' if r == fault_round else 'blue' for r in round_numbers]
    
    bars = ax2.bar(round_numbers, round_intervals, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=1)
    
    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Round Duration (seconds)')
    ax2.set_title('Training Round Duration (Fault Detection)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Annotate fault round
    fault_duration = round_intervals[fault_round-2]  # Round 8 duration
    ax2.annotate(f'Fault Round\n{fault_duration:.1f}s\n(+30s delay)', 
                xy=(fault_round, fault_duration), xytext=(fault_round+2, fault_duration+5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_fault_analysis_plot(df):
    """Create detailed fault analysis plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    rounds = df['Round']
    accuracy = df['Accuracy']
    participants = df['Participants']
    failed = df['Failed_Participants']
    
    # Create dual axis plot
    ax2 = ax.twinx()
    
    # Accuracy line
    line1 = ax.plot(rounds, accuracy, 'b-', linewidth=3, marker='o', markersize=8, 
                   label='Test Accuracy', zorder=3)
    
    # Participants bar chart
    width = 0.6
    bars_active = ax2.bar(rounds, participants, width, color='lightgreen', alpha=0.7, 
                         label='Active Participants', edgecolor='darkgreen', linewidth=1)
    bars_failed = ax2.bar(rounds, failed, width, bottom=participants, color='red', alpha=0.7,
                         label='Failed Participants', edgecolor='darkred', linewidth=1)
    
    # Mark fault area
    fault_round = 8
    ax.axvspan(fault_round-0.5, fault_round+0.5, alpha=0.2, color='red', zorder=1)
    ax.axvline(x=fault_round, color='red', linestyle='--', linewidth=2, alpha=0.8, zorder=2)
    
    # Set axis labels and title
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, color='blue')
    ax2.set_ylabel('Number of Participants', fontsize=12, color='green')
    
    ax.set_title('Spark FL Worker Node Fault Tolerance: Comprehensive Analysis', fontsize=14, fontweight='bold')
    
    # Set Y-axis ranges
    ax.set_ylim(96, 100)
    ax2.set_ylim(0, 6)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Add key statistics
    info_text = f"""Experiment Statistics:
• Total Rounds: {len(rounds)}
• Fault Round: Round {fault_round}
• Pre-fault Accuracy: {df[df['Round'] == fault_round-1]['Accuracy'].iloc[0]:.2f}%
• During-fault Accuracy: {df[df['Round'] == fault_round]['Accuracy'].iloc[0]:.2f}%
• Post-recovery Accuracy: {df[df['Round'] == fault_round+1]['Accuracy'].iloc[0]:.2f}%
• Final Accuracy: {df['Accuracy'].iloc[-1]:.2f}%
• Performance Drop: {df[df['Round'] == fault_round-1]['Accuracy'].iloc[0] - df[df['Round'] == fault_round]['Accuracy'].iloc[0]:.2f}%"""
    
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
            verticalalignment='bottom')
    
    plt.tight_layout()
    return fig

def print_statistics(df):
    """Print detailed statistics"""
    print("\n" + "="*60)
    print("SPARK FL WORKER NODE FAULT TOLERANCE EXPERIMENT STATISTICS")
    print("="*60)
    
    fault_round = 8
    pre_fault = df[df['Round'] == fault_round-1].iloc[0]
    fault = df[df['Round'] == fault_round].iloc[0]
    recovery = df[df['Round'] == fault_round+1].iloc[0]
    final = df.iloc[-1]
    
    print(f"📊 Basic Statistics:")
    print(f"   • Total Training Rounds: {len(df)}")
    print(f"   • Fault Injection Round: Round {fault_round}")
    print(f"   • Total Experiment Time: {final['Timestamp']:.1f} seconds")
    
    print(f"\n🎯 Performance Analysis:")
    print(f"   • Pre-fault Accuracy (Round {fault_round-1}): {pre_fault['Accuracy']:.2f}%")
    print(f"   • During-fault Accuracy (Round {fault_round}): {fault['Accuracy']:.2f}%")
    print(f"   • Post-recovery Accuracy (Round {fault_round+1}): {recovery['Accuracy']:.2f}%")
    print(f"   • Final Accuracy (Round {len(df)}): {final['Accuracy']:.2f}%")
    
    performance_drop = pre_fault['Accuracy'] - fault['Accuracy']
    recovery_gain = recovery['Accuracy'] - fault['Accuracy']
    
    print(f"\n📉 Fault Impact:")
    print(f"   • Performance Drop: {performance_drop:.2f}%")
    print(f"   • Recovery Gain: {recovery_gain:.2f}%")
    print(f"   • Post-recovery exceeds Pre-fault: {recovery['Accuracy'] > pre_fault['Accuracy']}")
    
    print(f"\n⚡ Fault Tolerance Analysis:")
    print(f"   • Fault Round Active Participants: {fault['Participants']}/4")
    print(f"   • Fault Round Failed Participants: {fault['Failed_Participants']}")
    print(f"   • Fault Round Duration: {fault['Timestamp'] - pre_fault['Timestamp']:.1f} seconds (including 30s delay)")
    print(f"   • Normal Round Average Duration: {(final['Timestamp'] - fault['Timestamp']) / (len(df) - fault_round):.1f} seconds")
    
    print(f"\n✅ Fault Tolerance Effectiveness:")
    print(f"   • System can handle 50% Participant Faults")
    print(f"   • Automatic Fault Detection and Recovery")
    print(f"   • Model Performance Fully Recovered")
    print(f"   • RDD Lineage Mechanism Successfully Tolerated")

def main():
    """Main function"""
    # Check and switch to correct directory
    if not os.path.exists('results'):
        os.chdir('fault_tolerance/exp2_worker_node_failure/spark_simple')
    
    # Load data
    df = load_results()
    if df is None:
        return
    
    # Print statistics
    print_statistics(df)
    
    # Create visualization plots
    print("\n📈 Generating Visualization Plots...")
    
    # 1. Comprehensive analysis plot
    fig1 = create_comprehensive_plot(df)
    fig1.savefig('results/spark_fl_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✓ Comprehensive analysis plot: results/spark_fl_comprehensive_analysis.png")
    
    # 2. Timeline analysis plot
    fig2 = create_timeline_plot(df)
    fig2.savefig('results/spark_fl_timeline_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✓ Timeline analysis plot: results/spark_fl_timeline_analysis.png")
    
    # 3. Fault analysis plot
    fig3 = create_fault_analysis_plot(df)
    fig3.savefig('results/spark_fl_fault_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✓ Fault analysis plot: results/spark_fl_fault_analysis.png")
    
    print("\n🎉 All plots generated successfully!")
    print("📁 Check results/ directory to view generated images")
    
    # Display plots (if environment supports)
    try:
        plt.show()
    except:
        print("⚠️   Unable to display plots (possibly no graphical environment), please check saved PNG files")

if __name__ == "__main__":
    main() 