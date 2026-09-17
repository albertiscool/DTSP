"""
=============================================================================
50 回合隨機地圖蒙地卡羅泛化性評估 (50-Run Monte Carlo Benchmark: V3 vs V5)
=============================================================================
評估 Version 3 (切線流平滑) 與 Version 5 (Transformer + 嚴格物理控制器)
在 50 個完全不同的隨機海域地圖（隨機種子 1~50）下的泛化性表現：
- Stage 0 (25 艘隨機靜態敵艦)
- Stage 5 (30 艘隨機敵艦，包含 5 艘動態出現的新敵艦)

評估指標：
1. 航程長度 (Trajectory Length: 平均值、標準差、中位數)
2. 碰撞次數 (Collision Count: 平均次數、0 碰撞完美成功率 %)
3. 計算耗時 (Computation Time: 平均耗時)
=============================================================================
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import concurrent.futures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 加入模組路徑
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("dtsp_uav"))

from compare_versions_v5 import Version3_TangentSmoothing, Version5_ZeroCollisionFast


def evaluate_single_seed(seed, max_iter=800):
    """在給定的隨機種子下評估 V3 與 V5 的 Stage 0 與 Stage 5 表現"""
    try:
        np.random.seed(seed)
        random.seed(seed)
        
        # 1. 生成 25 艘靜態敵艦 (50~150km, 0~100km)
        n_static = 25
        static_centers = np.column_stack([
            np.random.uniform(50, 150, n_static),
            np.random.uniform(0, 100, n_static)
        ])
        
        # 最西北角為基地起點
        norm_x = (static_centers[:, 0] - 50.0) / 100.0
        norm_y = (static_centers[:, 1] - 0.0) / 100.0
        start_node_idx = int(np.argmin(norm_x - norm_y))
        
        # 預先生成 5 個動態目標
        dynamic_targets = [
            np.random.uniform([50, 0], [150, 100]) for _ in range(5)
        ]
        total_centers_stage5 = np.vstack([static_centers] + dynamic_targets)
        
        # 實例化求解器 (強制使用 CPU 避免多行程 CUDA 衝突)
        v3_solver = Version3_TangentSmoothing()
        
        v5_solver = Version5_ZeroCollisionFast()
        v5_solver.device = torch.device('cpu')
        if v5_solver.policy is not None:
            v5_solver.policy = v5_solver.policy.to('cpu')
            
        # ---------------------------------------------------------------------
        # 1. Version 3 評估
        # ---------------------------------------------------------------------
        # Stage 0
        t0 = time.time()
        _, _, v3_len_0, v3_col_0 = v3_solver.solve(static_centers, start_node_idx, max_iter=max_iter)
        v3_time_0 = time.time() - t0
        
        # Stage 5
        t0 = time.time()
        _, _, v3_len_5, v3_col_5 = v3_solver.solve(total_centers_stage5, start_node_idx, max_iter=max_iter)
        v3_time_5 = time.time() - t0
        
        # ---------------------------------------------------------------------
        # 2. Version 5 評估
        # ---------------------------------------------------------------------
        # Stage 0 (固定種子保證一致性)
        random.seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        _, _, v5_len_0, v5_col_0 = v5_solver.solve(static_centers, start_node_idx, max_iter=max_iter)
        v5_time_0 = time.time() - t0
        
        # Stage 5
        random.seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        _, _, v5_len_5, v5_col_5 = v5_solver.solve(total_centers_stage5, start_node_idx, max_iter=max_iter)
        v5_time_5 = time.time() - t0
        
        return {
            'seed': seed,
            'v3_len_0': v3_len_0,
            'v3_col_0': v3_col_0,
            'v3_time_0': v3_time_0,
            'v3_len_5': v3_len_5,
            'v3_col_5': v3_col_5,
            'v3_time_5': v3_time_5,
            'v5_len_0': v5_len_0,
            'v5_col_0': v5_col_0,
            'v5_time_0': v5_time_0,
            'v5_len_5': v5_len_5,
            'v5_col_5': v5_col_5,
            'v5_time_5': v5_time_5,
        }
    except Exception as e:
        print(f"[ERROR] Seed {seed} failed: {e}")
        return None


def run_monte_carlo_benchmark(num_seeds=50, max_workers=8):
    print("=================================================================")
    print(f" [MONTE CARLO] Starting {num_seeds} Random Map Benchmark (V3 vs V5)")
    print(f" Multiprocessing Workers: {max_workers}")
    print("=================================================================")
    
    seeds = list(range(1, num_seeds + 1))
    results = []
    
    t_start = time.time()
    completed_count = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_seed = {executor.submit(evaluate_single_seed, s): s for s in seeds}
        
        for future in concurrent.futures.as_completed(future_to_seed):
            res = future.result()
            if res is not None:
                results.append(res)
                completed_count += 1
                s = res['seed']
                print(f"[{completed_count:02d}/{num_seeds:02d}] Seed {s:02d} | "
                      f"Stage 0: V3 ({res['v3_len_0']:.1f}km, {res['v3_col_0']} col) vs V5 ({res['v5_len_0']:.1f}km, {res['v5_col_0']} col) | "
                      f"Stage 5: V3 ({res['v3_len_5']:.1f}km, {res['v3_col_5']} col) vs V5 ({res['v5_len_5']:.1f}km, {res['v5_col_5']} col)")
                
    total_benchmark_time = time.time() - t_start
    print(f"\n[SUCCESS] Completed {len(results)} seeds in {total_benchmark_time:.2f} seconds ({total_benchmark_time/60.0:.2f} mins).")
    
    # 轉為 DataFrame 並儲存 CSV
    df = pd.DataFrame(results).sort_values(by='seed').reset_index(drop=True)
    csv_path = "monte_carlo_50_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] Results saved to {csv_path}")
    
    # 計算統計摘要
    print("\n" + "="*70)
    print(" [SUMMARY] 50 Maps Benchmark Statistical Summary Table")
    print("="*70)
    
    summary_data = {
        "Metric": [
            "Stage 0 平均航程 (km)",
            "Stage 0 航程標準差 (km)",
            "Stage 0 平均碰撞次數",
            "Stage 0 零碰撞率 (0-Col Rate)",
            "Stage 0 平均耗時 (s)",
            "---------------------------",
            "Stage 5 平均航程 (km)",
            "Stage 5 航程標準差 (km)",
            "Stage 5 平均碰撞次數",
            "Stage 5 零碰撞率 (0-Col Rate)",
            "Stage 5 平均耗時 (s)"
        ],
        "V3 (Tangent Smoothing)": [
            f"{df['v3_len_0'].mean():.2f}",
            f"{df['v3_len_0'].std():.2f}",
            f"{df['v3_col_0'].mean():.2f}",
            f"{(df['v3_col_0'] == 0).mean() * 100:.1f}%",
            f"{df['v3_time_0'].mean():.2f}s",
            "---------------------------",
            f"{df['v3_len_5'].mean():.2f}",
            f"{df['v3_len_5'].std():.2f}",
            f"{df['v3_col_5'].mean():.2f}",
            f"{(df['v3_col_5'] == 0).mean() * 100:.1f}%",
            f"{df['v3_time_5'].mean():.2f}s"
        ],
        "V5 (Transformer Strict)": [
            f"{df['v5_len_0'].mean():.2f}",
            f"{df['v5_len_0'].std():.2f}",
            f"{df['v5_col_0'].mean():.2f}",
            f"{(df['v5_col_0'] == 0).mean() * 100:.1f}%",
            f"{df['v5_time_0'].mean():.2f}s",
            "---------------------------",
            f"{df['v5_len_5'].mean():.2f}",
            f"{df['v5_len_5'].std():.2f}",
            f"{df['v5_col_5'].mean():.2f}",
            f"{(df['v5_col_5'] == 0).mean() * 100:.1f}%",
            f"{df['v5_time_5'].mean():.2f}s"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("="*70)
    
    # 繪製統計對比大圖
    plot_monte_carlo_charts(df)


def plot_monte_carlo_charts(df):
    """繪製 50 回合蒙地卡羅視覺化統計圖表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200)
    
    # 配色方案
    c_v3 = '#3b82f6'  # 科技藍
    c_v5 = '#10b981'  # 護眼綠
    
    # --- 圖 1: Stage 0 與 Stage 5 航程箱型圖 (Boxplot) ---
    ax1 = axes[0, 0]
    box_data = [df['v3_len_0'], df['v5_len_0'], df['v3_len_5'], df['v5_len_5']]
    box_labels = ['V3 (Stage 0)', 'V5 (Stage 0)', 'V3 (Stage 5)', 'V5 (Stage 5)']
    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.55,
                     medianprops=dict(color='#0f172a', linewidth=2),
                     boxprops=dict(facecolor='white', alpha=0.85))
    
    colors = [c_v3, c_v5, c_v3, c_v5]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        
    ax1.set_title("1. Trajectory Length Distribution (50 Random Maps)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Trajectory Length (km)", fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 標記平均值文字
    means = [d.mean() for d in box_data]
    for i, m in enumerate(means):
        ax1.text(i + 1, m + 15, f"Avg:\n{m:.1f}km", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1e293b')

    # --- 圖 2: 零碰撞成功率 (Zero-Collision Success Rate %) ---
    ax2 = axes[0, 1]
    v3_rate_0 = (df['v3_col_0'] == 0).mean() * 100
    v5_rate_0 = (df['v5_col_0'] == 0).mean() * 100
    v3_rate_5 = (df['v3_col_5'] == 0).mean() * 100
    v5_rate_5 = (df['v5_col_5'] == 0).mean() * 100
    
    x = np.arange(2)
    w = 0.35
    b1 = ax2.bar(x - w/2, [v3_rate_0, v3_rate_5], w, label='V3 Tangent Smoothing', color='#93c5fd', edgecolor='#1d4ed8')
    b2 = ax2.bar(x + w/2, [v5_rate_0, v5_rate_5], w, label='V5 Transformer Strict', color='#6ee7b7', edgecolor='#047857')
    
    ax2.set_title("2. Zero-Collision Success Rate (%)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("0-Collision Maps Rate (%)", fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Stage 0 (25 Targets)', 'Stage 5 (30 Targets)'], fontweight='bold')
    ax2.set_ylim(0, 115)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in b1 + b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, h + 2, f"{h:.1f}%", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    # --- 圖 3: 平均碰撞次數 (Average Collision Count) ---
    ax3 = axes[1, 0]
    v3_col_mean_0 = df['v3_col_0'].mean()
    v5_col_mean_0 = df['v5_col_0'].mean()
    v3_col_mean_5 = df['v3_col_5'].mean()
    v5_col_mean_5 = df['v5_col_5'].mean()
    
    c1 = ax3.bar(x - w/2, [v3_col_mean_0, v3_col_mean_5], w, label='V3 Tangent Smoothing', color='#fca5a5', edgecolor='#dc2626')
    c2 = ax3.bar(x + w/2, [v5_col_mean_0, v5_col_mean_5], w, label='V5 Transformer Strict', color='#a7f3d0', edgecolor='#059669')
    
    ax3.set_title("3. Average Collision Count (Lower is Better)", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Average Collisions per Map", fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Stage 0 (25 Targets)', 'Stage 5 (30 Targets)'], fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in c1 + c2:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, h + 0.05, f"{h:.2f}", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    # --- 圖 4: 50 回合航程差直方分佈 (V5 vs V3 Delta Length) ---
    ax4 = axes[1, 1]
    diff_stage0 = df['v3_len_0'] - df['v5_len_0']
    diff_stage5 = df['v3_len_5'] - df['v5_len_5']
    
    ax4.hist(diff_stage0, bins=15, alpha=0.6, label='Stage 0: (V3 - V5 Length)', color='#3b82f6', edgecolor='#1d4ed8')
    ax4.hist(diff_stage5, bins=15, alpha=0.6, label='Stage 5: (V3 - V5 Length)', color='#10b981', edgecolor='#047857')
    ax4.axvline(0, color='#ef4444', linestyle='--', linewidth=1.5, label='Zero Improvement Threshold')
    
    ax4.set_title("4. Length Improvement: V3 Length - V5 Length (km)", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Length Saved by V5 (km) [>0 means V5 is shorter]", fontweight='bold')
    ax4.set_ylabel("Number of Maps", fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("DTSP Generalization Benchmark: 50 Random Maps (V3 vs V5)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    
    out_fig = "monte_carlo_50_evaluation.png"
    plt.savefig(out_fig, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SUCCESS] Saved Benchmark Visualization: {out_fig}")


if __name__ == '__main__':
    run_monte_carlo_benchmark(num_seeds=50, max_workers=10)
