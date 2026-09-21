import os
import sys
import time
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 將路徑加入
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("dtsp_uav"))

from dtsp_uav.core.dubins import DubinsCost
from compare_versions_v4 import Version2_SmartInit, Version3_TangentSmoothing, Version4_TransformerAndSA, draw_single_panel

def main():
    FIXED_SEED = 42
    np.random.seed(FIXED_SEED)
    random.seed(FIXED_SEED)
    
    n_static = 25
    static_centers = np.column_stack([
        np.random.uniform(50, 150, n_static),
        np.random.uniform(0, 100, n_static)
    ])
    
    norm_x = (static_centers[:, 0] - 50) / 100.0
    norm_y = (static_centers[:, 1] - 0) / 100.0
    start_node_idx = int(np.argmin(norm_x - norm_y))
    
    v2_solver = Version2_SmartInit()
    v3_solver = Version3_TangentSmoothing()
    v4_solver = Version4_TransformerAndSA()
    cost_calc = DubinsCost(2.0)
    
    # 執行 V2
    t0 = time.time()
    v2_idx, v2_pts, v2_len, v2_col = v2_solver.solve(static_centers, start_node_idx)
    v2_time = time.time() - t0
    
    # 執行 V3
    t0 = time.time()
    v3_idx, v3_pts, v3_len, v3_col = v3_solver.solve(static_centers, start_node_idx)
    v3_time = time.time() - t0
    
    # 執行 V4
    t0 = time.time()
    v4_idx, v4_pts, v4_len, v4_col = v4_solver.solve(static_centers, start_node_idx)
    v4_time = time.time() - t0
    
    print(f"V2: {v2_len:.2f} km, {v2_col} cols, {v2_time:.2f}s")
    print(f"V3: {v3_len:.2f} km, {v3_col} cols, {v3_time:.2f}s")
    print(f"V4: {v4_len:.2f} km, {v4_col} cols, {v4_time:.2f}s")
    
    # 繪製 1x3 橫向高清大圖 (16:9 簡報首選)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5), dpi=220)
    fig.suptitle("DTSP Algorithm Comparison: Version 2 vs Version 3 vs Version 4 (Transformer)", 
                 fontsize=18, fontweight='bold', y=0.98)
    
    draw_single_panel(axes[0], static_centers, v2_idx, v2_pts, start_node_idx,
                      "Version 2: 2-Opt & Penalty (Heuristic SA)",
                      f"Length: {v2_len:.2f} km | Collisions: {v2_col} | Time: {v2_time:.2f}s", cost_calc)
                      
    draw_single_panel(axes[1], static_centers, v3_idx, v3_pts, start_node_idx,
                      "Version 3: Tangent Smoothing (Geometric SA)",
                      f"Length: {v3_len:.2f} km | Collisions: {v3_col} | Time: {v3_time:.2f}s", cost_calc)
                      
    draw_single_panel(axes[2], static_centers, v4_idx, v4_pts, start_node_idx,
                      "Version 4: Transformer + Fast SA (Deep Learning)",
                      f"Length: {v4_len:.2f} km | Collisions: {v4_col} | Time: {v4_time:.2f}s", cost_calc)
                      
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    plt.subplots_adjust(wspace=0.15)
    
    out_path = "comparison_v2_v3_v4.png"
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SUCCESS] Saved {out_path}")

if __name__ == "__main__":
    main()
