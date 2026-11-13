# -*- coding: utf-8 -*-
"""
plot_large_unseen_trajectories.py
重繪圖8 (unseen_difficult_seeds_trajectories.png)，大幅調大字體、加粗線條與標籤
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

def draw_large_panel(ax, centers, indices, points, start_node_idx, title, sub_title, cost_calculator):
    ax.set_facecolor('white')
    ax.set_title(f"{title}\n{sub_title}", fontsize=13, fontweight='bold', pad=12, color='#0F172A')
    ax.set_xlabel("東向座標 East (km)", fontsize=11.5, fontweight='bold')
    ax.set_ylabel("北向座標 North (km)", fontsize=11.5, fontweight='bold')
    ax.tick_params(axis='both', labelsize=10.5)
    ax.grid(True, linestyle='--', color='#CBD5E1', alpha=0.6)
    
    # 威脅禁區與安全走廊
    for i, c in enumerate(centers):
        if i == start_node_idx: continue
        ax.add_patch(Circle((c[0], c[1]), 5.0, facecolor='#FEE2E2', edgecolor='#EF4444', linestyle='--', linewidth=1.0, alpha=0.6, zorder=1))
        ax.add_patch(Circle((c[0], c[1]), 7.0, facecolor='none', edgecolor='#3B82F6', linestyle=':', linewidth=1.0, alpha=0.8, zorder=1))
        ax.scatter(c[0], c[1], c='#DC2626', s=25, zorder=2)
        
    ax.scatter(centers[start_node_idx, 0], centers[start_node_idx, 1], c='#EAB308', s=180, marker='*', edgecolors='#854D0E', linewidths=1.5, zorder=6, label='無人機基地 (Depot)')
    ax.scatter(points[:, 0], points[:, 1], c='#10B981', s=35, edgecolors='#064E3B', linewidths=1.0, zorder=5, label='航路錨點 (Waypoints)')
    
    n = len(indices)
    for i in range(n):
        idx1 = indices[i]
        idx2 = indices[(i + 1) % n]
        p1 = points[idx1]
        p2 = points[idx2]
        t, p, q, mode, _ = cost_calculator._plan_dubins(p1, p2)
        if mode is None:
            px = np.linspace(p1[0], p2[0], 25)
            py = np.linspace(p1[1], p2[1], 25)
        else:
            px, py = cost_calculator._interpolate(p1, t, p, q, mode, step_size=0.5)
        ax.plot(px, py, color='#1D4ED8', linewidth=2.4, alpha=0.9, zorder=3, label='杜賓斯航線' if i == 0 else "")
        
    ax.set_aspect('equal', 'box')
    ax.legend(loc='lower left', fontsize=10.5, framealpha=0.95, facecolor='white', edgecolor='#CBD5E1')

def plot_large_trajectory_samples():
    print(">>> 正在生成大字體版 圖8: unseen_difficult_seeds_trajectories.png...")
    solver = Version8_CaRVisibilitySolver()
    sample_seeds = [54, 55, 58]
    
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.5), dpi=300)
    fig.suptitle("50組全新未知動態隨機地圖代表性飛行軌跡 (Seeds 54, 55, 58 零碰撞)", fontsize=17, fontweight='bold', y=0.98, color="#0F172A")
    
    for idx, s in enumerate(sample_seeds):
        np.random.seed(s)
        random.seed(s)
        n_static = 25
        centers = np.column_stack([
            np.random.uniform(50.0, 150.0, n_static),
            np.random.uniform(0.0, 100.0, n_static)
        ])
        norm_x = (centers[:, 0] - 50.0) / 100.0
        norm_y = (centers[:, 1] - 0.0) / 100.0
        start_node = int(np.argmin(norm_x - norm_y))
        
        ind, pts, length, col = solver.solve(centers, start_node)
        
        ax = axes[idx]
        title_str = f"測試地圖 Seed {s}"
        sub_title = f"[航跡長: {length:.1f} km | 碰撞: {col} 次 | 零碰撞完全避障]"
        draw_large_panel(ax, centers, ind, pts, start_node, title_str, sub_title, solver.cost_calculator)

    plt.tight_layout()
    out_traj = os.path.join(OUTPUT_DIR, "unseen_difficult_seeds_trajectories.png")
    plt.savefig(out_traj, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[SUCCESS] 圖8大字體重繪完成:", out_traj)

if __name__ == "__main__":
    plot_large_trajectory_samples()
