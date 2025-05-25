#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traditional FL EXP2 Results Visualization (English Version)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

def load_results():
    """Load Traditional FL experiment results"""
    results_file = 'results/traditional/checkpoints/results.csv'
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return None
    
    df = pd.read_csv(results_file)
    print(f"Loaded data: {len(df)} rounds")
    return df

def create_comprehensive_plot(df):
    """Create comprehensive analysis plot for Traditional FL"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Traditional FL Worker Node Fault Tolerance Results', fontsize=16, fontweight='bold')
    
    rounds = df['Round']
    
    # 1. Accuracy trend plot
    ax1.plot(rounds, df['Accuracy'], 'g-', linewidth=2, marker='o', markersize=6, label='Test Accuracy')
    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy Progression')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(90, 100)
    
    # 2. Loss trend plot
    ax2.plot(rounds, df['Loss'], 'r-', linewidth=2, marker='s', markersize=6, label='Training Loss')
    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss Progression')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Training time analysis
    if len(df) > 1:
        time_intervals = []
        for i in range(1, len(df)):
            interval = df.iloc[i]['Timestamp'] - df.iloc[i-1]['Timestamp']
            time_intervals.append(interval)
        
        round_numbers = rounds[1:]
        ax3.bar(round_numbers, time_intervals, alpha=0.7, color='blue', edgecolor='black')
        ax3.set_xlabel('Training Round')
        ax3.set_ylabel('Round Duration (seconds)')
        ax3.set_title('Training Round Duration')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Performance summary
    if len(df) >= 3:
        initial_acc = df.iloc[0]['Accuracy']
        mid_acc = df.iloc[len(df)//2]['Accuracy']
        final_acc = df.iloc[-1]['Accuracy']
        
        summary_metrics = {
            'Initial\n(Round 1)': initial_acc,
            'Mid-training\n(Round {})'.format(len(df)//2 + 1): mid_acc,
            'Final\n(Round {})'.format(len(df)): final_acc
        }
        
        bars = ax4.bar(summary_metrics.keys(), summary_metrics.values(), 
                       color=['lightcoral', 'lightblue', 'lightgreen'],
                       alpha=0.8, edgecolor='black', linewidth=1)
        
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Performance Summary')
        ax4.set_ylim(90, 100)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, summary_metrics.values()):
            ax4.annotate(f'{value:.2f}%', 
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_comparison_plot(df):
    """Create detailed performance analysis plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Traditional FL Performance Analysis', fontsize=16, fontweight='bold')
    
    rounds = df['Round']
    timestamps = df['Timestamp']
    
    # 1. Accuracy vs Time
    ax1.plot(timestamps, df['Accuracy'], 'g-', linewidth=3, marker='o', markersize=8, 
             markerfacecolor='white', markeredgecolor='green', markeredgewidth=2)
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Performance Timeline')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(90, 100)
    
    # 2. Round progression with timing
    if len(df) > 1:
        round_intervals = []
        for i in range(1, len(df)):
            interval = df.iloc[i]['Timestamp'] - df.iloc[i-1]['Timestamp']
            round_intervals.append(interval)
        
        round_numbers = rounds[1:]
        bars = ax2.bar(round_numbers, round_intervals, alpha=0.7, color='orange', 
                       edgecolor='black', linewidth=1)
        
        ax2.set_xlabel('Training Round')
        ax2.set_ylabel('Round Duration (seconds)')
        ax2.set_title('Training Efficiency Analysis')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add average line
        avg_duration = np.mean(round_intervals)
        ax2.axhline(y=avg_duration, color='red', linestyle='--', linewidth=2, 
                   label=f'Average: {avg_duration:.1f}s')
        ax2.legend()
    
    plt.tight_layout()
    return fig

def print_statistics(df):
    """Print detailed statistics"""
    print("\n" + "="*60)
    print("TRADITIONAL FL EXPERIMENT STATISTICS")
    print("="*60)
    
    print(f"📊 Basic Statistics:")
    print(f"   • Total Training Rounds: {len(df)}")
    print(f"   • Total Experiment Time: {df.iloc[-1]['Timestamp']:.1f} seconds")
    print(f"   • Initial Accuracy: {df.iloc[0]['Accuracy']:.2f}%")
    print(f"   • Final Accuracy: {df.iloc[-1]['Accuracy']:.2f}%")
    print(f"   • Accuracy Improvement: {df.iloc[-1]['Accuracy'] - df.iloc[0]['Accuracy']:.2f}%")
    
    if len(df) > 1:
        round_intervals = []
        for i in range(1, len(df)):
            interval = df.iloc[i]['Timestamp'] - df.iloc[i-1]['Timestamp']
            round_intervals.append(interval)
        
        print(f"\n⏱️ Timing Analysis:")
        print(f"   • Average Round Duration: {np.mean(round_intervals):.1f} seconds")
        print(f"   • Fastest Round: {np.min(round_intervals):.1f} seconds")
        print(f"   • Slowest Round: {np.max(round_intervals):.1f} seconds")
        print(f"   • Standard Deviation: {np.std(round_intervals):.1f} seconds")
    
    print(f"\n📈 Performance Analysis:")
    print(f"   • Peak Accuracy: {df['Accuracy'].max():.2f}%")
    print(f"   • Lowest Accuracy: {df['Accuracy'].min():.2f}%")
    print(f"   • Final Loss: {df.iloc[-1]['Loss']:.4f}")
    print(f"   • Accuracy Variance: {df['Accuracy'].var():.4f}")

def main():
    """Main function"""
    # Check and switch to correct directory
    if not os.path.exists('results'):
        print("Current directory doesn't have results folder, switching...")
        os.chdir('fault_tolerance/exp2_worker_node_failure/simple_run')
    
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
    os.makedirs('results/traditional/plots', exist_ok=True)
    fig1.savefig('results/traditional/plots/traditional_fl_comprehensive_analysis.png', 
                 dpi=300, bbox_inches='tight')
    print("   ✓ Comprehensive analysis plot: results/traditional/plots/traditional_fl_comprehensive_analysis.png")
    
    # 2. Performance comparison plot
    fig2 = create_comparison_plot(df)
    fig2.savefig('results/traditional/plots/traditional_fl_performance_analysis.png', 
                 dpi=300, bbox_inches='tight')
    print("   ✓ Performance analysis plot: results/traditional/plots/traditional_fl_performance_analysis.png")
    
    print("\n🎉 All plots generated successfully!")
    print("📁 Check results/traditional/plots/ directory to view generated images")
    
    # Display plots (if environment supports)
    try:
        plt.show()
    except:
        print("⚠️   Unable to display plots (possibly no graphical environment), please check saved PNG files")

if __name__ == "__main__":
    main() 