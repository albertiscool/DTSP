"""
=============================================================================
論文方法單次推論（One-Shot）性能對比實驗基準
Benchmark One-Shot Comparison on Unseen Seeds (51 ~ 100)
=============================================================================
對比文獻中所提出的三大核心改進方案在完全未見過海域（Seeds 51~100）上的單次機會（One-Shot）零碰撞率：

方法 1 [Baseline]: 標準單次推論 (V5 Standard One-Shot)
   - 傳統神經啟發式 + 靜態幾何外推 + 盲目均勻退火 (Blind SA)

方法 2 [Method A - Corridor/Visibility Graph]: 可見度走廊拓撲增強 (IEEE T-RO 2023)
   - 強化走廊安全閾值 (Corridor Clearance 6.8 km) + 宏觀序列拓撲修復

方法 3 [Method B - Analytical Bitangents]: 解析外公切線對齊 (Robotics & Auto. Systems)
   - 針對近接雙目標 (< 8.5 km) 採用外公切線與弦向航向平滑化，消除自碰撞

方法 4 [Method C - CaR Integrated]: 整合式構建-修復框架 (NeurIPS 2023 / ICLR 2024 CaR)
   - 走廊拓撲過濾 + 焦點可行性投影 (Focused Projection) + 自適應切線幾何橡皮擦
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

from compare_versions_v5 import Version5_ZeroCollisionFast, DubinsCost
from optimize_to_100 import AdvancedV5Optimizer

# 日誌輸出
LOG_FILE = "benchmark_oneshot_comparison.log"
def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")
        f.flush()

# Worker 全域求解器快取
_v5_baseline = None
_v5_advanced = None

def get_solvers():
    global _v5_baseline, _v5_advanced
    if _v5_baseline is None:
        _v5_baseline = Version5_ZeroCollisionFast()
        _v5_baseline.device = torch.device('cpu')
        if _v5_baseline.policy: _v5_baseline.policy = _v5_baseline.policy.to('cpu')
    if _v5_advanced is None:
        _v5_advanced = AdvancedV5Optimizer()
        _v5_advanced.device = torch.device('cpu')
        if _v5_advanced.policy: _v5_advanced.policy = _v5_advanced.policy.to('cpu')
    return _v5_baseline, _v5_advanced


def evaluate_seed_methods(seed):
    try:
        np.random.seed(seed)
        random.seed(seed)
        
        # 1. 生成海域環境
        n_static = 25
        centers = np.column_stack([
            np.random.uniform(50, 150, n_static),
            np.random.uniform(0, 100, n_static)
        ])
        norm_x = (centers[:, 0] - 50.0) / 100.0
        norm_y = (centers[:, 1] - 0.0) / 100.0
        start_node_idx = int(np.argmin(norm_x - norm_y))
        
        v5_base, v5_adv = get_solvers()
        
        # -------------------------------------------------------------
        # 評測 1: Baseline One-Shot (原版 V5 盲退火單次推論)
        # -------------------------------------------------------------
        t0 = time.time()
        random.seed(seed); np.random.seed(seed)
        _, _, l_base, col_base = v5_base.solve(centers, start_node_idx, max_iter=450)
        time_base = time.time() - t0
        
        # -------------------------------------------------------------
        # 評測 2: Method A (IEEE T-RO 可見度走廊拓撲增強, 無自適應橡皮擦)
        # -------------------------------------------------------------
        t0 = time.time()
        random.seed(seed); np.random.seed(seed)
        _, _, l_corridor, col_corridor = v5_adv.solve_enhanced(
            centers, start_node_idx, strategy="standard", max_iter=450, corridor_thresh=6.8
        )
        time_corridor = time.time() - t0
        
        # -------------------------------------------------------------
        # 評測 3: Method B/C (NeurIPS CaR 整合式框架: 走廊過濾 + 焦點可行性投影 + 自適應切線橡皮擦)
        # -------------------------------------------------------------
        t0 = time.time()
        random.seed(seed); np.random.seed(seed)
        _, _, l_car, col_car = v5_adv.solve_enhanced(
            centers, start_node_idx, strategy="adaptive_radius", max_iter=450, corridor_thresh=6.8
        )
        time_car = time.time() - t0
        
        return {
            'seed': seed,
            'base_col': col_base,
            'base_len': l_base,
            'base_time': time_base,
            'corridor_col': col_corridor,
            'corridor_len': l_corridor,
            'corridor_time': time_corridor,
            'car_col': col_car,
            'car_len': l_car,
            'car_time': time_car,
        }
    except Exception as e:
        log(f"[ERROR] Seed {seed} failed: {e}")
        return None


def main():
    log("=" * 80)
    log(" 論文方法單次推論（One-Shot）泛化性全面對比實驗啟動")
    log(" 測試對象：隨機種子 51 ~ 100（50 組全新完全未見過隨機海域）")
    log(" 測試約束：嚴格 One-Shot 推論（單次機會，無任何多起點/多種子重新隨機重啟）")
    log(" 運行環境：6 核心進程池 (ProcessPoolExecutor, max_workers=6)")
    log("=" * 80)
    
    seeds = list(range(51, 101))
    t_start = time.time()
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(evaluate_seed_methods, s): s for s in seeds}
        completed = 0
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res is not None:
                results.append(res)
            completed += 1
            if completed % 10 == 0 or completed == 50:
                log(f"   [進度] 已完成 {completed}/50 組種子盲測對比...")
                
    df = pd.DataFrame(results).sort_values(by='seed').reset_index(drop=True)
    df.to_csv("oneshot_comparison_results.csv", index=False)
    
    total_time = time.time() - t_start
    log("=" * 80)
    log(f"【評測完成】總耗時: {total_time:.1f} 秒 ({total_time/60:.1f} 分鐘)")
    log("=" * 80)
    
    n_total = len(df)
    base_zero = (df['base_col'] == 0).sum()
    corr_zero = (df['corridor_col'] == 0).sum()
    car_zero = (df['car_col'] == 0).sum()
    
    log(f"【單次推論（One-Shot）核心指標對比報告 (Unseen Seeds 51~100)】")
    log(f" 1. Baseline (原版 V5 盲退火單次推論):")
    log(f"    - 零碰撞率 (Zero-Collision %): {base_zero}/{n_total} ({base_zero/n_total*100.1:.1f}%)")
    log(f"    - 平均航程長度 (Avg Length):    {df['base_len'].mean():.2f} km")
    log(f"    - 平均推論時間 (Avg Time):      {df['base_time'].mean():.2f} 秒")
    log("-" * 80)
    log(f" 2. Method A (IEEE T-RO 可見度走廊拓撲增強):")
    log(f"    - 零碰撞率 (Zero-Collision %): {corr_zero}/{n_total} ({corr_zero/n_total*100.1:.1f}%)")
    log(f"    - 平均航程長度 (Avg Length):    {df['corridor_len'].mean():.2f} km")
    log(f"    - 平均推論時間 (Avg Time):      {df['corridor_time'].mean():.2f} 秒")
    log("-" * 80)
    log(f" 3. Method B/C (NeurIPS CaR 構建-修復框架: 走廊過濾 + 焦點可行性投影 + 自適應切線橡皮擦):")
    log(f"    - 零碰撞率 (Zero-Collision %): {car_zero}/{n_total} ({car_zero/n_total*100.1:.1f}%)")
    log(f"    - 平均航程長度 (Avg Length):    {df['car_len'].mean():.2f} km")
    log(f"    - 平均推論時間 (Avg Time):      {df['car_time'].mean():.2f} 秒")
    log("=" * 80)

if __name__ == "__main__":
    main()
