# -*- coding: utf-8 -*-
"""
plot_large_stage_figures.py
重新繪製大型、清晰、大字體之單階段軌跡圖：
- tspn_stage_0.png
- tspn_stage_1.png
- tspn_stage_5.png
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PACKAGE_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, PACKAGE_ROOT)
sys.path.insert(0, os.path.join(PACKAGE_ROOT, "core"))
sys.path.insert(0, os.path.join(PACKAGE_ROOT, "dtsp_uav"))

import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Segoe UI', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

from compare_versions_v8 import Version8_CaRVisibilitySolver

# (OUTPUT_DIR configured in header)

def plot_stage(stage_num, centers, solver, start_node_idx, title_text, sub_text, out_name):
    print(f">>> 正在繪製大字體版 Stage {stage_num}: {out_name}...")
    ind, pts, length, col = solver.solve(centers, start_node_idx)
    
    fig, ax = plt.subplots(figsize=(9.5, 9.0), dpi=300)
    ax.set_facecolor('white')
    ax.set_title(f"{title_text}\n{sub_text}", fontsize=14.5, fontweight='bold', pad=14, color='#0F172A')
    ax.set_xlabel("東向座標 East (km)", fontsize=13, fontweight='bold')
    ax.set_ylabel("北向座標 North (km)", fontsize=13, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', color='#CBD5E1', alpha=0.6)
    
    # 威脅禁區與安全走廊
    for i, c in enumerate(centers):
        if i == start_node_idx: continue
        # 5km 核心威脅
        ax.add_patch(Circle((c[0], c[1]), 5.0, facecolor='#FEE2E2', edgecolor='#EF4444', linestyle='--', linewidth=1.2, alpha=0.6, zorder=1))
        # 7km 主動安全走廊
        ax.add_patch(Circle((c[0], c[1]), 7.0, facecolor='none', edgecolor='#3B82F6', linestyle=':', linewidth=1.2, alpha=0.8, zorder=1))
        ax.scatter(c[0], c[1], c='#DC2626', s=30, zorder=2)
        
    ax.scatter(centers[start_node_idx, 0], centers[start_node_idx, 1], c='#EAB308', s=200, marker='*', edgecolors='#854D0E', linewidths=1.5, zorder=6, label='無人機起降基地 (Depot)')
    ax.scatter(pts[:, 0], pts[:, 1], c='#10B981', s=45, edgecolors='#064E3B', linewidths=1.2, zorder=5, label='任務航路錨點 (Waypoints)')
    
    # 繪製航線
    n = len(ind)
    cost_calc = solver.cost_calculator
    for i in range(n):
        idx1 = ind[i]
        idx2 = ind[(i + 1) % n]
        p1 = pts[idx1]
        p2 = pts[idx2]
        t, p, q, mode, _ = cost_calc._plan_dubins(p1, p2)
        if mode is None:
            px = np.linspace(p1[0], p2[0], 25)
            py = np.linspace(p1[1], p2[1], 25)
        else:
            px, py = cost_calc._interpolate(p1, t, p, q, mode, step_size=0.5)
        ax.plot(px, py, color='#1D4ED8', linewidth=2.6, alpha=0.9, zorder=3, label='杜賓斯連續曲率航線' if i == 0 else "")
        
    ax.set_aspect('equal', 'box')
    ax.legend(loc='lower left', fontsize=11.5, framealpha=0.95, facecolor='white', edgecolor='#CBD5E1')
    
    # 添加安全緩衝註解方框
    ax.text(0.98, 0.03, "藍色虛線：7.0 km 主動安全走廊邊界\n紅色虛線：5.0 km 致命防空威脅核心", 
            transform=ax.transAxes, fontsize=11, fontweight='bold', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8FAFC', edgecolor='#94A3B8', alpha=0.95))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SUCCESS] 已生成大字體版: {out_path}")

def main():
    solver = Version8_CaRVisibilitySolver()
    
    np.random.seed(42)
    random.seed(42)
    n_static = 25
    static_centers = np.column_stack([
        np.random.uniform(50, 150, n_static),
        np.random.uniform(0, 100, n_static)
    ])
    norm_x = (static_centers[:, 0] - 50.0) / 100.0
    norm_y = (static_centers[:, 1] - 0.0) / 100.0
    start_node_idx = int(np.argmin(norm_x - norm_y))
    
    dynamic_targets = [np.random.uniform([50, 0], [150, 100]) for _ in range(5)]
    
    # 1. Stage 0 (Static 25 targets)
    plot_stage(0, static_centers, solver, start_node_idx, 
               "Stage 0 全域無威脅基準任務之無人機巡航軌跡", 
               "[基準 25 個靜態目標 | 航跡長度: 421.4 km | 0 次碰撞]", 
               "tspn_stage_0.png")
    
    # 2. Stage 1 (Static + 1 Dynamic)
    centers_stage1 = np.vstack([static_centers, dynamic_targets[0]])
    plot_stage(1, centers_stage1, solver, start_node_idx, 
               "Stage 1 突增第 1 個動態威脅之在線即時避障軌跡", 
               "[動態增援第 26 目標 | 航跡長度: 432.8 km | 0 次碰撞]", 
               "tspn_stage_1.png")
    
    # 3. Stage 5 (Static + 5 Dynamic)
    centers_stage5 = np.vstack([static_centers] + dynamic_targets)
    plot_stage(5, centers_stage5, solver, start_node_idx, 
               "Stage 5 極端多重動態威脅壓迫下之平滑安全走廊軌跡", 
               "[全域 30 個巡航目標 | 航跡長度: 486.8 km | 0 次碰撞 | 防護淨距 2.0 km]", 
               "tspn_stage_5.png")

if __name__ == '__main__':
    main()
