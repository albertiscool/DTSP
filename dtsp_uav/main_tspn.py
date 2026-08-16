import numpy as np
import random
import os
from core.environment import Environment
from core.dubins import DubinsCost
from core.routing import Route
from algorithms.simulated_annealing_tspn import SimulatedAnnealingTSPN
from simulation.animator_tspn import AnimatorTSPN

def build_initial_euclidean_tour(centers, start_idx):
    """使用貪婪最近鄰 (Nearest Neighbor) + 2-opt 構建無自交的歐氏初始閉環"""
    unvisited = set(range(len(centers)))
    unvisited.remove(start_idx)
    tour = [start_idx]
    curr = start_idx
    while unvisited:
        next_node = min(unvisited, key=lambda idx: np.linalg.norm(centers[curr] - centers[idx]))
        tour.append(next_node)
        unvisited.remove(next_node)
        curr = next_node
    
    # 執行快速 2-opt 解開歐氏交叉線
    improved = True
    n = len(tour)
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                d0 = np.linalg.norm(centers[a] - centers[b]) + np.linalg.norm(centers[c] - centers[d])
                d1 = np.linalg.norm(centers[a] - centers[c]) + np.linalg.norm(centers[b] - centers[d])
                if d1 < d0 - 1e-4:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
    return tour

def insert_new_center(route_indices, new_center_idx, centers):
    """
    使用最便宜插入啟發式 (Cheapest Insertion) 將新敵艦中心插入到現有路徑中。
    基於歐氏距離進行快速預估。
    """
    best_pos = None
    best_increase = float('inf')
    n = len(route_indices)
    
    for i in range(n):
        idx_a = route_indices[i]
        idx_b = route_indices[(i + 1) % n]
        
        d_a_new = np.linalg.norm(centers[idx_a] - centers[new_center_idx])
        d_new_b = np.linalg.norm(centers[new_center_idx] - centers[idx_b])
        d_a_b = np.linalg.norm(centers[idx_a] - centers[idx_b])
        
        increase = d_a_new + d_new_b - d_a_b
        if increase < best_increase:
            best_increase = increase
            best_pos = i + 1
            
    updated_indices = list(route_indices)
    updated_indices.insert(best_pos, new_center_idx)
    
    # 動態插入後執行快速 2-opt 消除交叉
    improved = True
    n = len(updated_indices)
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = updated_indices[i - 1], updated_indices[i]
                c, d = updated_indices[j], updated_indices[(j + 1) % n]
                d0 = np.linalg.norm(centers[a] - centers[b]) + np.linalg.norm(centers[c] - centers[d])
                d1 = np.linalg.norm(centers[a] - centers[c]) + np.linalg.norm(centers[b] - centers[d])
                if d1 < d0 - 1e-4:
                    updated_indices[i:j + 1] = updated_indices[i:j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
    return updated_indices

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
        cooling_rate=0.996,
        max_iter=3000
    )
    
    # 尋找左上角起點 (Depot / Launch Base)
    start_node_idx = env.get_start_node_index()
    print(f"UAV 起飛基地索引: {start_node_idx}，位置: {centers[start_node_idx]}")

    # 建立高品質初始歐氏閉環順序 (消除初始交叉)
    route_indices = build_initial_euclidean_tour(centers, start_node_idx)
    route = Route(route_indices)

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
        
        # 透過 Cheapest Insertion 先插入到 sequence
        updated_indices = insert_new_center(route.indices, new_center_idx, centers)
        route = Route(updated_indices)
        
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
