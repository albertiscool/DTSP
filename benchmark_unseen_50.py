"""
=============================================================================
全新 50 組隨機種子泛化性驗證 (Unseen 50-Seed Generalization Benchmark)
=============================================================================
測試範圍：隨機種子 51 ~ 100（模型在先前所有優化過程中完全未見過的全新海域）
測試目標：
1. 驗證 Stage 0（25 靜態目標）的零碰撞完美率 %
2. 驗證 Stage 5（30 動態目標）的零碰撞完美率 %
3. 統計平均航程長度 (km) 與平均推論耗時 (s)
4. 檢查是否存在過擬合現象（Overfitting Test）
=============================================================================
"""

import os
import sys
import time
import random
import datetime
import numpy as np
import pandas as pd
import torch
import concurrent.futures

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("dtsp_uav"))

from compare_versions_v5 import Version5_ZeroCollisionFast

# UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

LOG_FILE = "benchmark_unseen_50.log"

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")
        f.flush()

# Worker solver cache
_worker_solver = None
def get_solver():
    global _worker_solver
    if _worker_solver is None:
        _worker_solver = Version5_ZeroCollisionFast()
        _worker_solver.device = torch.device('cpu')
        if _worker_solver.policy is not None:
            _worker_solver.policy = _worker_solver.policy.to('cpu')
    return _worker_solver

def evaluate_single_unseen_seed(seed, max_iter=500):
    try:
        np.random.seed(seed)
        random.seed(seed)
        
        # 1. 生成全新海域 25 艘靜態敵艦
        n_static = 25
        static_centers = np.column_stack([
            np.random.uniform(50, 150, n_static),
            np.random.uniform(0, 100, n_static)
        ])
        
        norm_x = (static_centers[:, 0] - 50.0) / 100.0
        norm_y = (static_centers[:, 1] - 0.0) / 100.0
        start_node_idx = int(np.argmin(norm_x - norm_y))
        
        # 預先生成 5 個動態目標
        dynamic_targets = [
            np.random.uniform([50, 0], [150, 100]) for _ in range(5)
        ]
        total_centers_stage5 = np.vstack([static_centers] + dynamic_targets)
        
        solver = get_solver()
        
        # Stage 0 單次標準推論
        t0 = time.time()
        random.seed(seed); np.random.seed(seed)
        _, _, s0_len, s0_col = solver.solve(static_centers, start_node_idx, max_iter=max_iter)
        s0_time = time.time() - t0
        
        # Stage 5 單次標準推論
        t0 = time.time()
        random.seed(seed); np.random.seed(seed)
        _, _, s5_len, s5_col = solver.solve(total_centers_stage5, start_node_idx, max_iter=max_iter)
        s5_time = time.time() - t0
        
        return {
            'seed': seed,
            's0_len': s0_len,
            's0_col': s0_col,
            's0_time': s0_time,
            's5_len': s5_len,
            's5_col': s5_col,
            's5_time': s5_time,
        }
    except Exception as e:
        log(f"[ERROR] Seed {seed} failed: {e}")
        return None

def main():
    log("=" * 70)
    log(" 全新 50 組未曾見過的隨機種子 (Seed 51 ~ 100) 泛化性盲測啟動")
    log(" 測試環境: 6 核心並行加速 (ProcessPoolExecutor, Workers=6)")
    log("=" * 70)
    
    seeds = list(range(51, 101))
    t_start = time.time()
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(evaluate_single_unseen_seed, s): s for s in seeds}
        completed = 0
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res is not None:
                results.append(res)
            completed += 1
            if completed % 10 == 0 or completed == 50:
                log(f"   [進度] 已完成 {completed}/50 個全新種子評估...")
                
    df_unseen = pd.DataFrame(results).sort_values(by='seed').reset_index(drop=True)
    df_unseen.to_csv("unseen_50_results.csv", index=False)
    
    dur = time.time() - t_start
    log("=" * 70)
    log(f"【盲測完成】耗時: {dur:.1f} 秒 ({dur/60:.1f} 分鐘)")
    
    s0_zero = (df_unseen['s0_col'] == 0).sum()
    s5_zero = (df_unseen['s5_col'] == 0).sum()
    s0_rate = s0_zero / len(df_unseen) * 100.0
    s5_rate = s5_zero / len(df_unseen) * 100.0
    
    log(f" 盲測評估報告 (Unseen Seeds 51~100):")
    log(f" - Stage 0 零碰撞率: {s0_zero}/{len(df_unseen)} ({s0_rate:.1f}%) | 平均航程: {df_unseen['s0_len'].mean():.2f} km | 平均耗時: {df_unseen['s0_time'].mean():.2f} s")
    log(f" - Stage 5 零碰撞率: {s5_zero}/{len(df_unseen)} ({s5_rate:.1f}%) | 平均航程: {df_unseen['s5_len'].mean():.2f} km | 平均耗時: {df_unseen['s5_time'].mean():.2f} s")
    log("=" * 70)

if __name__ == "__main__":
    main()
