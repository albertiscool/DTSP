import numpy as np
import random
from core.routing import Route

class SimulatedAnnealing:
    def __init__(self, cost_fn, initial_temp=None, cooling_rate=0.995, max_iter=2000):
        """
        初始化模擬退火演算法。
        :param initial_temp: 初始溫度，若為 None 則啟動適應性溫度計算
        :param cooling_rate: 冷卻速率
        :param max_iter: 最大迭代次數
        """
        self.cost_fn = cost_fn
        self.initial_temp = initial_temp  
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter

    def _get_combined_cost(self, indices, headings, points):
        """計算結合特定航向的路徑總成本"""
        total = 0.0
        for i in range(len(indices)):
            u, v = indices[i], indices[(i + 1) % len(indices)]
            p1 = np.append(points[u], headings[u])
            p2 = np.append(points[v], headings[v])
            total += self.cost_fn(p1, p2)
        return total

    def _estimate_initial_temperature(self, indices, headings, points, samples=100, target_prob=0.8):
        """
        透過樣本擾動估計初始溫度，使初始接受率接近 target_prob。
        """
        deltas = []
        n = len(indices)
        current_cost = self._get_combined_cost(indices, headings, points)
        
        for _ in range(samples):
            # 隨機選擇一種擾動
            if random.random() < 0.5:
                idx1, idx2 = random.sample(range(1, n), 2)
                if idx1 > idx2: idx1, idx2 = idx2, idx1
                new_idx = indices[:idx1] + indices[idx1:idx2+1][::-1] + indices[idx2+1:]
                new_h = headings
            else:
                new_idx = indices
                new_h = np.array(headings)
                target = random.randint(0, n - 1)
                new_h[target] = (new_h[target] + np.random.normal(0, np.radians(20))) % (2 * np.pi)
            
            new_cost = self._get_combined_cost(new_idx, new_h, points)
            deltas.append(abs(new_cost - current_cost))
        
        avg_delta = np.mean(deltas) if deltas else 1.0
        # 根據 exp(-delta/T) = P -> T = -delta / ln(P)
        return -avg_delta / np.log(target_prob)

    def optimize(self, route, points):
        """
        使用模擬退火演算法優化路徑。
        採用 2-opt 鄰域算子，並保持路徑起點（索引 0）固定。
        :param route: 初始 Route 物件
        :param points: 目標點座標矩陣
        :return: 優化後的 Route 物件
        """
        n = len(route)
        if n < 4:
            return route

        # 初始狀態：索引順序與隨機航向 (或初始為指向下一個點的方向)
        current_indices = list(route.indices)
        current_headings = np.zeros(len(points)) 
        
        # 初始航向設為指向下一個目標點，這是一個比全 0 更好的初始值
        for i in range(n):
            p1 = points[current_indices[i]]
            p2 = points[current_indices[(i+1)%n]]
            current_headings[current_indices[i]] = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])

        current_cost = self._get_combined_cost(current_indices, current_headings, points)
        
        best_indices = list(current_indices)
        best_headings = np.array(current_headings)
        best_cost = current_cost
        
        # 適應性溫度計算
        temp = self.initial_temp if self.initial_temp is not None else \
               self._estimate_initial_temperature(current_indices, current_headings, points)
        print(f"適應性初始溫度設定為: {temp:.4f}")

        for i in range(self.max_iter):
            new_indices = list(current_indices)
            new_headings = np.array(current_headings)

            # 隨機選擇擾動策略
            rand_val = random.random()
            if rand_val < 0.4:
                # 1. 2-opt 順序改變
                idx1, idx2 = random.sample(range(1, n), 2)
                if idx1 > idx2: idx1, idx2 = idx2, idx1
                new_indices = current_indices[:idx1] + current_indices[idx1:idx2+1][::-1] + current_indices[idx2+1:]
            elif rand_val < 0.8:
                # 2. 航向擾動：隨機選一個點改變其進入角度
                target_idx = random.randint(0, n - 1)
                new_headings[target_idx] += np.random.normal(0, np.radians(15)) # 擾動 15 度
                new_headings[target_idx] %= (2 * np.pi)
            else:
                # 3. 智慧航向對齊：將點的航向對齊到「前一點」與「後一點」的向量方向 (減少繞路關鍵)
                target_pos = random.randint(0, n - 1)
                target_node = current_indices[target_pos]
                prev_node = current_indices[(target_pos - 1) % n]
                next_node = current_indices[(target_pos + 1) % n]
                
                # 計算從前一點指向後一點的向量角度
                vec_x = points[next_node, 0] - points[prev_node, 0]
                vec_y = points[next_node, 1] - points[prev_node, 1]
                new_headings[target_node] = np.arctan2(vec_y, vec_x)

            new_cost = self._get_combined_cost(new_indices, new_headings, points)
            
            # Metropolis 接受準則
            delta = new_cost - current_cost
            if delta < 0 or (temp > 0 and random.random() < np.exp(-delta / temp)):
                current_indices = new_indices
                current_headings = new_headings
                current_cost = new_cost
                
                # 更新全局最佳解
                if current_cost < best_cost:
                    best_indices = list(current_indices)
                    best_headings = np.array(current_headings)
                    best_cost = current_cost
            
            # 降溫
        
        # 將優化後的航向寫回 points 矩陣 (或是回傳包含航向的座標)
        optimized_points = np.column_stack((points[:, 0], points[:, 1], best_headings))
        return Route(best_indices), optimized_points