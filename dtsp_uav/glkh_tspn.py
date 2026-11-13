import numpy as np
import random
import os
import subprocess
from core.environment import Environment
from core.dubins import DubinsCost
from core.routing import Route
from algorithms.simulated_annealing_tspn import SimulatedAnnealingTSPN
from simulation.animator_tspn import AnimatorTSPN

# Apply
def write_glkh_file(points, filename="problem.tsp"):
    """將座標寫入 TSPLIB 格式，供 GLKH 使用"""
    n = len(points)
    with open(filename, "w") as f:
        f.write("NAME : TSP_Task\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, p in enumerate(points):
            f.write(f"{i+1} {p[0]:.4f} {p[1]:.4f}\n")
        f.write("EOF\n")

# Apply
def run_glkh(executable_path, problem_file):
    """呼叫 C 語言核心求解器"""
    print(f"啟動 GLKH 求解器 (核心: {executable_path})...")
    try:
        result = subprocess.run([executable_path, problem_file], 
                                capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"執行失敗，回傳碼: {result.returncode}")
            print(f"錯誤訊息: {result.stderr}")
            return None
        return result.stdout
    except Exception as e:
        print(f"執行失敗: {e}")
        return None

# Apply
def parse_glkh_output(num_points):
    """解析 GLKH 輸出的路徑順序。"""
    indices = list(range(num_points)) # 預設回傳原始順序作為 fallback
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

def solve_glkh(centers, start_node_idx, glkh_executable="/home/ee720a/DTSP/dtsp_uav/GLKH", tsp_file="problem.tsp"):
    """使用 GLKH 求解 TSP，回傳 Route"""
    write_glkh_file(centers, tsp_file)
    output = run_glkh(glkh_executable, tsp_file)
    if output is None:
        raise RuntimeError("GLKH 執行失敗")
    if not os.path.exists("problem.tour"):
        raise RuntimeError("找不到 problem.tour")
    
    indices = parse_glkh_output(len(centers))
    idx = indices.index(start_node_idx)
    indices = indices[idx:] + indices[:idx]
    
    return Route(indices)

def main():
    # 設定亂數種子，使地圖敵艦中心點與原 DTSP 一致，以便進行長度對比
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    print("=== Dubins TSP with Neighborhoods (DTSPN) & Obstacle Avoidance ===")
    print("1. 初始化環境 (100x100 km, 25個初始敵艦目標)")
    env = Environment(x_range=(50, 150), y_range=(0, 100), n_static=25)
    centers = env.get_all_targets().copy() # 敵艦中心點座標
    
    # 參數設定
    turning_radius = 2.0   # 無人機轉彎半徑 2.0 km (同 main.py)
    obstacle_radius = 5.0  # 避障禁航半徑 5.0 km
    obs_min_radius = 5.0   # 觀測甜甜圈內徑 5.0 km
    obs_max_radius = 9.0   # 觀測甜甜圈外徑 9.0 km
    
    # 初始化優化器
    sa = SimulatedAnnealingTSPN(
        turning_radius=turning_radius,
        obstacle_radius=obstacle_radius,
        obs_min_radius=obs_min_radius,
        obs_max_radius=obs_max_radius,
        cooling_rate=0.998,
        max_iter=10000  # 增加迭代次數與減緩降溫速率以達到零碰撞與航程最佳化
    )
    
    # 尋找左上角起點 (Depot / Launch Base)
    start_node_idx = env.get_start_node_index()  # 修改這裡
    print(f"UAV 起飛基地索引: {start_node_idx}，位置: {centers[start_node_idx]}")

    # 使用 GLKH 求解初始 TSP
    print("GLKH 求解初始 TSP...")
    route = solve_glkh(centers, start_node_idx)
    
    # 2. 初始路徑優化 (DTSPN + 避障)
    print("優化初始 DTSPN 路徑中 (包含 24 艘敵艦的 5km 避障與 5-9km 觀測限制)...")
    route, points, phi_angles, radii = sa.optimize(route, centers, start_node_idx)
    
    # 紀錄快照
    snapshots = []
    snapshots.append((centers.copy(), points.copy(), Route(route.indices), start_node_idx, phi_angles.copy(), radii.copy()))
    
    # 計算初始 Dubins 長度與碰撞檢測
    _, final_col = sa._get_combined_cost_and_collisions(route.indices, points[:, 2], phi_angles, radii, centers, start_node_idx)
    cost_calculator = DubinsCost(turning_radius)
    closed_idx = route.get_closed_path()
    total_len = 0.0
    for j in range(len(closed_idx) - 1):
        u, v = closed_idx[j], closed_idx[j+1]
        total_len += cost_calculator.cost(points[u], points[v])
    print(f"初始規劃完成，總長度: {total_len:.2f} km，碰撞次數: {final_col}\n")

    # 3. 模擬動態加入 5 個不明敵艦目標
    for i in range(5):
        # 隨機生成一個新敵艦
        new_target = np.array([
            np.random.uniform(env.x_range[0], env.x_range[1]),
            np.random.uniform(env.y_range[0], env.y_range[1])
        ])
        print(f"--- 偵測到新不明敵艦 {i+1}: {new_target} ---")
        
        # 更新環境中的中心點
        centers = np.vstack([centers, new_target])
        new_center_idx = len(centers) - 1
        
        # 使用 GLKH 重新規劃 TSP
        print("GLKH 重新規劃 TSP...")
        route = solve_glkh(centers, start_node_idx)
        
        # 執行 SA 重規劃與航向/角度優化
        print(f"進行路徑重規劃與 DTSPN 避障優化...")
        route, points, phi_angles, radii = sa.optimize(route, centers, start_node_idx)
        
        # 紀錄快照
        snapshots.append((centers.copy(), points.copy(), Route(route.indices), start_node_idx, phi_angles.copy(), radii.copy()))
        
        # 計算長度與碰撞
        _, final_col = sa._get_combined_cost_and_collisions(route.indices, points[:, 2], phi_angles, radii, centers, start_node_idx)
        closed_idx = route.get_closed_path()
        total_len = 0.0
        for j in range(len(closed_idx) - 1):
            u, v = closed_idx[j], closed_idx[j+1]
            total_len += cost_calculator.cost(points[u], points[v])
        print(f"動態規劃階段 {i+1} 完成，總長度: {total_len:.2f} km，碰撞次數: {final_col}\n")

    # 4. 輸出最終結果與生成快照圖片
    print("=== 任務模擬完成 ===")
    closed_idx = route.get_closed_path()
    total_len = 0.0
    for j in range(len(closed_idx) - 1):
        u, v = closed_idx[j], closed_idx[j+1]
        total_len += cost_calculator.cost(points[u], points[v])
    print(f"最終路徑總長度: {total_len:.2f} km")
    
    # 渲染快照
    animator = AnimatorTSPN(env, turning_radius, obstacle_radius, obs_max_radius)
    animator.render_snapshots(snapshots, save_dir=".")

if __name__ == "__main__":
    main()