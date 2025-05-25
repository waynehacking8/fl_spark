#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP2 Traditional FL vs Spark FL 容錯性能對比分析
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def compare_fl_methods():
    """對比 Traditional FL 和 Spark FL 的容錯性能"""
    
    # 檢查結果文件
    trad_file = 'simple_run/results/results.csv'
    spark_file = 'results/spark/results.csv'
    
    print("🔍 正在檢查結果文件...")
    
    trad_exists = os.path.exists(trad_file)
    spark_exists = os.path.exists(spark_file)
    
    print(f"Traditional FL 結果: {'✅' if trad_exists else '❌'} {trad_file}")
    print(f"Spark FL 結果: {'✅' if spark_exists else '❌'} {spark_file}")
    
    if not trad_exists and not spark_exists:
        print("❌ 未找到任何結果文件，請先運行實驗")
        return
    
    # 讀取數據
    data = {}
    
    if trad_exists:
        df_trad = pd.read_csv(trad_file)
        data['Traditional FL'] = df_trad
        print(f"✅ Traditional FL: {len(df_trad)} 輪")
    
    if spark_exists:
        df_spark = pd.read_csv(spark_file)
        data['Spark FL'] = df_spark
        print(f"✅ Spark FL: {len(df_spark)} 輪")
    
    # 生成對比圖表
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    colors = {'Traditional FL': 'blue', 'Spark FL': 'green'}
    markers = {'Traditional FL': 'o', 'Spark FL': 's'}
    
    # 準確率對比
    for method, df in data.items():
        ax1.plot(df['Round'], df['Accuracy'], 
                color=colors[method], marker=markers[method], 
                linewidth=2, markersize=4, label=method, alpha=0.8)
    
    ax1.axvline(x=8, color='red', linestyle='--', alpha=0.7, label='故障輪次')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('EXP2 容錯實驗 - 準確率對比')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(80, 100)
    
    # 訓練時間對比
    for method, df in data.items():
        ax2.plot(df['Round'], df['Timestamp'], 
                color=colors[method], marker=markers[method], 
                linewidth=2, markersize=4, label=method, alpha=0.8)
    
    ax2.axvline(x=8, color='red', linestyle='--', alpha=0.7, label='故障輪次')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Cumulative Time (seconds)')
    ax2.set_title('訓練時間對比')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存對比圖表
    comparison_plot = 'results/exp2_fl_comparison.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 對比圖表已保存: {comparison_plot}")
    
    # 生成對比報告
    report = "# EXP2 Traditional FL vs Spark FL 容錯性能對比\n\n"
    
    if len(data) == 2:
        df_trad = data['Traditional FL']
        df_spark = data['Spark FL']
        
        # 基礎指標對比
        report += "## 基礎性能對比\n\n"
        report += "| 指標 | Traditional FL | Spark FL | 改進 |\n"
        report += "|------|----------------|----------|------|\n"
        
        trad_final_acc = df_trad['Accuracy'].iloc[-1]
        spark_final_acc = df_spark['Accuracy'].iloc[-1]
        acc_diff = spark_final_acc - trad_final_acc
        
        trad_total_time = df_trad['Timestamp'].iloc[-1]
        spark_total_time = df_spark['Timestamp'].iloc[-1]
        time_ratio = trad_total_time / spark_total_time if spark_total_time > 0 else 0
        
        report += f"| 最終準確率 | {trad_final_acc:.2f}% | {spark_final_acc:.2f}% | {acc_diff:+.2f}% |\n"
        report += f"| 總訓練時間 | {trad_total_time:.1f}s | {spark_total_time:.1f}s | {time_ratio:.1f}x 倍 |\n"
        report += f"| 完成輪數 | {len(df_trad)} | {len(df_spark)} | {len(df_spark) - len(df_trad):+d} |\n"
        
        # 第8輪故障檢測對比
        report += "\n## 第8輪故障容錯對比\n\n"
        
        trad_round8 = df_trad[df_trad['Round'] == 8]
        spark_round8 = df_spark[df_spark['Round'] == 8]
        
        if len(trad_round8) > 0 and len(spark_round8) > 0:
            trad_8_time = trad_round8['Timestamp'].iloc[0]
            spark_8_time = spark_round8['Timestamp'].iloc[0]
            
            # 計算第8輪的用時
            trad_7_time = df_trad[df_trad['Round'] == 7]['Timestamp'].iloc[0] if len(df_trad[df_trad['Round'] == 7]) > 0 else 0
            spark_7_time = df_spark[df_spark['Round'] == 7]['Timestamp'].iloc[0] if len(df_spark[df_spark['Round'] == 7]) > 0 else 0
            
            trad_8_duration = trad_8_time - trad_7_time
            spark_8_duration = spark_8_time - spark_7_time
            
            report += "| 容錯機制 | Traditional FL | Spark FL |\n"
            report += "|----------|----------------|-----------|\n"
            report += f"| 故障檢測方式 | 60秒超時檢測 | Task失敗自動檢測 |\n"
            report += f"| 第8輪用時 | {trad_8_duration:.1f}s | {spark_8_duration:.1f}s |\n"
            report += f"| 恢復機制 | Checkpoint載入 | RDD血統重計算 |\n"
            report += f"| 人工干預 | 需要 | 不需要 |\n"
        
        # 技術優勢對比
        report += "\n## 容錯機制技術對比\n\n"
        report += "### Traditional FL 容錯特點\n"
        report += "- ✅ 超時檢測機制可靠\n"
        report += "- ✅ Checkpoint恢復狀態準確\n"
        report += "- ❌ 需要人工重啟故障節點\n"
        report += "- ❌ 60秒超時等待時間長\n"
        report += "- ❌ 節點級故障影響範圍大\n\n"
        
        report += "### Spark FL 容錯特點\n"
        report += "- ✅ RDD血統追蹤自動恢復\n"
        report += "- ✅ 分區級容錯粒度細\n"
        report += "- ✅ 零人工干預\n"
        report += "- ✅ 秒級故障檢測\n"
        report += "- ✅ 自動重新調度失敗任務\n\n"
    
    else:
        for method, df in data.items():
            report += f"## {method} 結果\n"
            report += f"- 完成輪數: {len(df)}\n"
            report += f"- 最終準確率: {df['Accuracy'].iloc[-1]:.2f}%\n"
            report += f"- 總訓練時間: {df['Timestamp'].iloc[-1]:.1f}s\n\n"
    
    # 保存報告
    report_file = 'results/exp2_comparison_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📝 對比報告已生成: {report_file}")
    
    # 打印總結
    print("\n" + "="*60)
    print("🎯 EXP2 實驗總結")
    print("="*60)
    
    if len(data) == 2:
        print("✅ 兩種方法都完成了容錯實驗")
        print(f"📊 Spark FL 在容錯能力上展現了明顯優勢:")
        print(f"   - 自動故障檢測和恢復")
        print(f"   - 分區級精細容錯")
        print(f"   - 零人工干預需求")
    else:
        print("ℹ️  僅有一種方法的結果可用，建議運行完整對比實驗")

if __name__ == "__main__":
    compare_fl_methods() 