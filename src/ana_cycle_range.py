#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析同一层不同 Segments/Containers 配置下，跨所有 neuron 的周期统计：
– LAYER_INDEX: 要分析的层号
– SEG_MIN…SEG_MAX, CONT_MIN…CONT_MAX: Segments 和 Containers 的扫描范围
输出：
  • 控制台打印每个 (S, C) 的 min/max/mean/std/count
  • CSV 文件 layer_{LAYER_INDEX}_SC_stats.csv
"""

import os
import json
import numpy as np
import csv

# --------- 配置区 ---------
BASE_DIR      = 'archive/json_0623_k4_serial_prune'
LAYER_INDEX   = 0    # 要分析的层号
SEG_MIN, SEG_MAX   = 4, 12
CONT_MIN, CONT_MAX = 4, 12
# -------------------------

LAYER_DIR = os.path.join(BASE_DIR, f'layer_{LAYER_INDEX}')
OUTPUT_CSV = os.path.join(
    BASE_DIR, f'layer_{LAYER_INDEX}_SC_stats.csv'
)

def collect_cycle_summary(neuron_dir):
    summary_path = os.path.join(neuron_dir, 'cycle_summary.json')
    if not os.path.isfile(summary_path):
        return None
    with open(summary_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    if not os.path.isdir(LAYER_DIR):
        print(f"找不到目录：{LAYER_DIR}")
        return

    # 先载入所有 neuron 的 summary
    neuron_summaries = []
    for neuron_name in sorted(os.listdir(LAYER_DIR)):
        neuron_dir = os.path.join(LAYER_DIR, neuron_name)
        summary = collect_cycle_summary(neuron_dir)
        if summary is not None:
            neuron_summaries.append(summary)
    if not neuron_summaries:
        print("未找到任何 neuron 的 cycle_summary.json，退出。")
        return

    # 准备输出表头
    stats = []
    header = ['Segments', 'Containers', 'count', 'min', 'max', 'mean', 'std']

    # 遍历 S, C
    for S in range(SEG_MIN, SEG_MAX + 1):
        for C in range(CONT_MIN, CONT_MAX + 1):
            # 从每个 neuron summary 取值
            vals = []
            for summary in neuron_summaries:
                # summary 的键是字符串
                s_dict = summary.get(str(S), {})
                v = s_dict.get(str(C))
                if isinstance(v, (int, float)):
                    vals.append(v)
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            stats.append({
                'Segments': S,
                'Containers': C,
                'count':   len(arr),
                'min':     float(np.min(arr)),
                'max':     float(np.max(arr)),
                'mean':    float(np.mean(arr)),
                'std':     float(np.std(arr)),
            })

    # 控制台输出
    print(f"Layer {LAYER_INDEX} – Segments/Containers 跨 neuron 周期统计")
    print(f"{'S':>3} {'C':>3} {'cnt':>5} {'min':>8} {'max':>8} {'mean':>8} {'std':>8}")
    for row in stats:
        print(f"{row['Segments']:>3} {row['Containers']:>3} "
              f"{row['count']:>5} "
              f"{row['min']:>8.1f} {row['max']:>8.1f} "
              f"{row['mean']:>8.1f} {row['std']:>8.1f}")

    # 写入 CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=header)
        writer.writeheader()
        for row in stats:
            writer.writerow(row)

    print(f"\n统计结果已保存到：{OUTPUT_CSV}")

if __name__ == '__main__':
    main()
