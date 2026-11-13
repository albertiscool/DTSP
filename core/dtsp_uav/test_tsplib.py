import numpy as np
import os
from core.dubins import DubinsCost
from core.routing import Route
from algorithms.insertion import CheapestInsertion
from algorithms.simulated_annealing import SimulatedAnnealing
from simulation.animator import Animator

def load_tsplib_coords(filepath):
    """解析 TSPLIB 檔案格式取得座標"""
    coords = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        node_coord_start = False
        for line in lines:
            if "NODE_COORD_SECTION" in line:
                node_coord_start = True
                continue
            if "EOF" in line or not line.strip():
                if node_coord_start: break
                continue
            if node_coord_start:
                parts = line.split()
                # 格式通常為: [ID] [X] [Y]
                if len(parts) >= 3:
                    coords.append([float(parts[1]), float(parts[2])])
    return np.array(coords)

class MockEnv:
    """為了讓 Animator 正常運作而模擬的 Environment 物件"""
    def __init__(self, points):
        padding = 5
        self.x_range = (np.min(points[:, 0]) - padding, np.max(points[:, 0]) + padding)
        self.y_range = (np.min(points[:, 1]) - padding, np.max(points[:, 1]) + padding)
        self.targets = points

def main():
    # 1. 載入 Ulysses22 座標
    tsp_path = "tsplib/ulysses22.tsp"
    if not os.path.exists(tsp_path):
        print(f"找不到測試檔案: {tsp_path}")
        return

    points = load_tsplib_coords(tsp_path)
    print(f"成功載入 {len(points)} 個目標點。")

    # 2. 初始化 Dubins 成本 (轉彎半徑設為 0.5 以適應 Ulysses 座標尺度)
    turning_radius = 0.5
    cost_fn = DubinsCost(turning_radius=turning_radius)
    
    # 3. 尋找左上角起點
    # Ulysses 座標中，x 最小且 y 最大為左上角
    top_left_ref = np.array([np.min(points[:, 0]), np.max(points[:, 1])])
    distances = np.linalg.norm(points - top_left_ref, axis=1)
    start_node_idx = np.argmin(distances)
    print(f"起點定位成功：索引 {start_node_idx}, 座標 {points[start_node_idx]}")

    # 4. 執行 Cheapest Insertion (初始路徑)
    inserter = CheapestInsertion(cost_fn.cost)

    # 這裡的關鍵是建立一個「動態增長」的點集，使索引與 Route 一致
    # 首先將起點放入工作點集，其在工作集中的索引為 0
    working_points = points[start_node_idx].reshape(1, 2)
    route = Route([0])

    remaining_indices = [i for i in range(len(points)) if i != start_node_idx]
    for next_idx in remaining_indices:
        new_target = points[next_idx]
        working_points = np.vstack([working_points, new_target])
        # CheapestInsertion.insert 會抓取 working_points 的最後一個索引作為新點
        route = inserter.insert(route, new_target, working_points)

    baseline_cost = route.calculate_total_cost(working_points, cost_fn.cost)
    print(f"\n[Baseline] Cheapest Insertion 總航程: {baseline_cost:.2f} km")

    # 5. 執行 Simulated Annealing (優化路徑)
    # 自由 Heading 增加了搜尋空間，建議將迭代次數提升至 15,000 次以上
    # 並放慢冷卻速率 (0.998 -> 0.999) 以確保角度能有足夠時間微調
    sa = SimulatedAnnealing(cost_fn.cost, cooling_rate=0.9995, max_iter=30000)

    print("正在執行模擬退火優化...")
    optimized_route, working_points = sa.optimize(route, working_points)
    
    optimized_cost = optimized_route.calculate_total_cost(working_points, cost_fn.cost)
    improvement = (baseline_cost - optimized_cost) / baseline_cost * 100

    print(f"[Optimized] Simulated Annealing 總航程: {optimized_cost:.2f} km")
    print(f"優化效率: {improvement:.2f}%")

    # 6. 視覺化對比
    env = MockEnv(working_points)
    animator = Animator(env)
    animator.turning_radius = turning_radius
    
    print("\n展示優化後的最終路徑...")
    # 在 reordered 的 working_points 中，起點的索引始終是 0
    animator.render(working_points, optimized_route, 0)
    
    # 更改標題顯示優化結果
    animator.ax.set_title(f"TSPLIB Ulysses22 - Optimized Dubins Path\nCost: {optimized_cost:.2f} km (Imp: {improvement:.2f}%)")
    animator.show()

if __name__ == "__main__":
    main()