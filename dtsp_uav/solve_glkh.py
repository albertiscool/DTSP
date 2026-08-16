import numpy as np
import random
import subprocess
import os
import time
from core.environment import Environment
from core.dubins import DubinsCost
from core.routing import Route
from algorithms.insertion import CheapestInsertion
from algorithms.simulated_annealing import SimulatedAnnealing
from dynamic.replanner import Replanner
from simulation.animator import Animator

def write_glkh_file(points, filename="problem.tsp"):
    """將座標寫入 TSPLIB 格式，供 GLKH 使用"""
    n = len(points)
    with open(filename, "w") as f:
        f.write("NAME : DTSP_Task\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, p in enumerate(points):
            f.write(f"{i+1} {p[0]:.4f} {p[1]:.4f}\n")
        f.write("EOF\n")

def run_glkh(executable_path, problem_file):
    """呼叫 C 語言核心求解器"""
    # 這裡假設 GLKH 接受參數或讀取預設的 .par 檔
    # 如果你的 GLKH 有特定的參數需求，請在此調整命令列語法
    print(f"啟動 GLKH 求解器 (核心: {executable_path})...")
    
    # 範例：./GLKH problem.tsp (實際依據你的 C 核心介面調整)
    try:
        # 這裡使用簡單的 subprocess 呼叫
        result = subprocess.run([executable_path, problem_file], 
                                capture_output=True, text=True, timeout=30)
        return result.stdout
    except Exception as e:
        print(f"執行失敗: {e}")
        return None

def parse_glkh_output(output, num_points):
    """
    解析 GLKH 輸出的路徑順序。
    這是一個通用解析器，假設輸出包含類似 'TOUR_SECTION' 的內容。
    """
    # 註：如果 GLKH 直接寫入 .sol 檔案，則應改為讀取檔案
    # 這裡先手動模擬一個假設的解析邏輯
    # 實務上你可能需要讀取 GLKH 產生的 *.sol 或 *.tour 檔
    indices = list(range(num_points)) # 預設回傳原始順序作為 fallback
    
    # 如果有輸出檔案 (例如 problem.tour)，讀取它
    tour_file = "problem.tour"
    if os.path.exists(tour_file):
        with open(tour_file, "r") as f:
            lines = f.readlines()
            start_parse = False
            tour = []
            for line in lines:
                if "TOUR_SECTION" in line:
                    start_parse = True
                    continue
                if start_parse:
                    idx = int(line.strip())
                    if idx == -1: break
                    tour.append(idx - 1) # TSPLIB 是從 1 開始計數
            indices = tour
    return indices

def main():
    # 設定亂數種子，確保地圖與 main.py 一致
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # 1. 初始化環境 (與 main.py 相同)
    env = Environment(x_range=(50, 150), y_range=(0, 100), n_static=25)
    points = env.get_all_targets()
    
    # 2. 設定求解器組件
    turning_radius = 1.0
    cost_fn = DubinsCost(turning_radius=turning_radius)
    executable = "/home/ee720a/DTSP/dtsp_uav/GLKH" 
    
    inserter = CheapestInsertion(cost_fn.cost)
    replanner = Replanner(inserter)
    sa = SimulatedAnnealing(cost_fn.cost, max_iter=2000) # 用於優化 Heading

    start_node_idx = env.get_start_node_index()
    snapshots = []

    print(f"地圖已生成，種子: {seed}, 初始目標數: {len(points)}")

    if not os.path.exists(executable):
        print(f"錯誤: 找不到 GLKH 執行檔 {executable}，請先編譯。")
        return

    # 3. 初始規劃：使用 GLKH 獲取最佳順序，再用 SA 優化航向
    write_glkh_file(points, "problem.tsp")
    output = run_glkh(executable, "problem.tsp")
    optimized_indices = parse_glkh_output(output, len(points))

    if start_node_idx in optimized_indices:
        s_pos = optimized_indices.index(start_node_idx)
        optimized_indices = optimized_indices[s_pos:] + optimized_indices[:s_pos]
    
    route = Route(optimized_indices)
    # 透過 SA 進行航向微調 (GLKH 只負責 XY 順序)
    route, points = sa.optimize(route, points)
    
    snapshots.append((points.copy(), route, start_node_idx))
    print(f"初始規劃完成，總長度: {route.calculate_total_cost(points, cost_fn.cost):.2f} km")

    # 4. 模擬動態加入 5 個不明目標
    for i in range(5):
        new_target = np.array([
            np.random.uniform(env.x_range[0], env.x_range[1]),
            np.random.uniform(env.y_range[0], env.y_range[1])
        ])
        print(f"偵測到不明目標 {i+1}: {new_target}，呼叫 GLKH 重規劃順序...")
        
        # 更新點集 (處理 3D 座標相容性)
        target_to_add = new_target
        if points.shape[1] == 3:
            target_to_add = np.append(new_target, 0.0)
        
        # 先用 replanner 插入新點，再用 GLKH 徹底重新排列
        route, points = replanner.update(route, target_to_add, points)
        
        # 重新寫入檔案呼叫 GLKH
        write_glkh_file(points, "problem.tsp")
        output = run_glkh(executable, "problem.tsp")
        optimized_indices = parse_glkh_output(output, len(points))
        
        if start_node_idx in optimized_indices:
            s_pos = optimized_indices.index(start_node_idx)
            optimized_indices = optimized_indices[s_pos:] + optimized_indices[:s_pos]
        
        route = Route(optimized_indices)
        # 局部細化航向
        route, points = sa.optimize(route, points)
        
        snapshots.append((points.copy(), route, start_node_idx))

    print(f"任務完成。最終路徑總長度: {route.calculate_total_cost(points, cost_fn.cost):.2f} km")

    # 5. 視覺化結果
    animator = Animator(env)
    animator.render_snapshots(snapshots)

if __name__ == "__main__":
    main()