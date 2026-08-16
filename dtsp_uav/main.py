import numpy as np
import random
from core.environment import Environment
from core.dubins import DubinsCost
from core.routing import Route
from algorithms.insertion import CheapestInsertion
from algorithms.simulated_annealing import SimulatedAnnealing
from dynamic.replanner import Replanner
from simulation.animator import Animator

def main():
    # 設定亂數種子，固定地圖目標位置與演算法隨機性
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # 1. 初始化環境 (100x100 km, 25個初始目標)
    env = Environment(x_range=(50, 150), y_range=(0, 100), n_static=25)
    points = env.get_all_targets()
    
    # 2. 設定 Dubins 成本計算與重新規劃器
    # 假設無人機在 120km/h 下的轉彎半徑約為 1km (可根據物理限制調整)
    turning_radius = 2.0 
    cost_fn = DubinsCost(turning_radius=turning_radius)
    inserter = CheapestInsertion(cost_fn.cost)
    replanner = Replanner(inserter)

    # 3. 尋找左上角起點並建立初始路徑
    start_node_idx = env.get_start_node_index()
    
    # 將起點排在第一位，其餘點暫按索引排序以形成初始閉環
    initial_indices = list(range(len(points)))
    initial_indices.remove(start_node_idx)
    route_indices = [start_node_idx] + initial_indices
    route = Route(route_indices)

    # 3.1 使用模擬退火優化初始路徑 (提升航程效率)
    sa = SimulatedAnnealing(cost_fn.cost, max_iter=2000)
    print("優化初始路徑中...")
    route, points = sa.optimize(route, points)

    # 4. 初始化動畫器並收集快照
    animator = Animator(env)
    animator.turning_radius = turning_radius
    snapshots = []

    # 紀錄初始狀態 (0 個動態目標)
    snapshots.append((points.copy(), route, start_node_idx))

    print(f"初始規劃完成，起點索引: {start_node_idx}，總長度: {route.calculate_total_cost(points, cost_fn.cost):.2f} km")

    # 5. 模擬動態加入 5 個不明目標
    for i in range(5):
        # 隨機生成一個區域內的新目標
        new_target = np.array([
            np.random.uniform(env.x_range[0], env.x_range[1]),
            np.random.uniform(env.y_range[0], env.y_range[1])
        ])
        print(f"偵測到不明目標 {i+1}: {new_target}，進行路徑重規劃...")
        
        # 確保新目標與目前帶有航向的 points 矩陣維度相符 (3D)
        target_to_add = new_target
        if points.shape[1] == 3:
            target_to_add = np.append(new_target, 0.0)

        route, points = replanner.update(route, target_to_add, points)
        # 動態插入後進行局部細化優化
        route, points = sa.optimize(route, points)
        
        # 紀錄快照
        snapshots.append((points.copy(), route, start_node_idx))

    print(f"任務完成。最終路徑總長度: {route.calculate_total_cost(points, cost_fn.cost):.2f} km")
    animator.render_snapshots(snapshots)

if __name__ == "__main__":
    main()