import numpy as np
import random
from core.dubins import DubinsCost
from core.routing import Route

class SimulatedAnnealingTSPN:
    def __init__(self, turning_radius=2.0, obstacle_radius=5.0, obs_min_radius=5.0, obs_max_radius=9.0, cooling_rate=0.995, max_iter=3000):
        """
        DTSPN 模擬退火優化器。
        優化目標包含：
        1. 訪問順序 (Sequence)
        2. 在甜甜圈觀測帶 (Annulus) 的拜訪點座標 (radius, phi)
        3. 拜訪時的無人機航向角 (heading)
        同時避開 5km 的敵艦中心障礙物。
        """
        self.turning_radius = turning_radius
        self.obstacle_radius = obstacle_radius
        self.obs_min_radius = obs_min_radius
        self.obs_max_radius = obs_max_radius
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.cost_calculator = DubinsCost(turning_radius)

    def _get_combined_cost_and_collisions(self, indices, headings, phi_angles, radii, centers, start_node_idx):
        """
        計算特定配置的總成本。
        成本 = 總 Dubins 航線長度 + 碰撞次數 * 1000.0 (懲罰權重)
        """
        n_points = len(centers)
        points = np.zeros((n_points, 3))
        for i in range(n_points):
            if i == start_node_idx:
                points[i, 0] = centers[i, 0]
                points[i, 1] = centers[i, 1]
            else:
                r = radii[i]
                phi = phi_angles[i]
                points[i, 0] = centers[i, 0] + r * np.cos(phi)
                points[i, 1] = centers[i, 1] + r * np.sin(phi)
            points[i, 2] = headings[i]

        total_len = 0.0
        num_collisions = 0
        n_route = len(indices)
        
        penalty_base = 2000.0   # 每次碰撞的基本懲罰
        penalty_slope = 3000.0  # 每侵入 1 km 額外增加的懲罰
        collision_cost = 0.0
        
        for i in range(n_route):
            u = indices[i]
            v = indices[(i + 1) % n_route]
            
            p1 = points[u]
            p2 = points[v]
            
            # 計算 Dubins 航線
            t, p, q, mode, length = self.cost_calculator._plan_dubins(p1, p2)
            if mode is None:
                dist = np.linalg.norm(p1[:2] - p2[:2])
                length = dist
                px = np.linspace(p1[0], p2[0], 20)
                py = np.linspace(p1[1], p2[1], 20)
            else:
                px, py = self.cost_calculator._interpolate(p1, t, p, q, mode, step_size=2.0)
            
            total_len += length
            
            # 避障檢查 (排除目前起終點，其餘敵艦為障礙物，且 launch depot 起點無障礙物)
            other_mask = np.ones(n_points, dtype=bool)
            other_mask[u] = False
            other_mask[v] = False
            other_mask[start_node_idx] = False
            
            obstacle_centers = centers[other_mask]
            if len(obstacle_centers) > 0:
                dx = px[:, np.newaxis] - obstacle_centers[np.newaxis, :, 0]
                dy = py[:, np.newaxis] - obstacle_centers[np.newaxis, :, 1]
                dists_sq = dx**2 + dy**2
                min_dists_sq = np.min(dists_sq, axis=0)
                
                for dist_sq in min_dists_sq:
                    if dist_sq < self.obstacle_radius**2:
                        num_collisions += 1
                        dist = np.sqrt(dist_sq)
                        depth = self.obstacle_radius - dist
                        collision_cost += penalty_base + penalty_slope * depth
            
            # 檢查是否進入起終點自身的 5km 障礙區 (使用 4.99km 避免端點浮點數誤差)
            if u != start_node_idx:
                du = np.sqrt((px - centers[u, 0])**2 + (py - centers[u, 1])**2)
                min_du = np.min(du)
                if min_du < 4.99:
                    num_collisions += 1
                    depth = 5.0 - min_du
                    collision_cost += penalty_base + penalty_slope * depth
            if v != start_node_idx:
                dv = np.sqrt((px - centers[v, 0])**2 + (py - centers[v, 1])**2)
                min_dv = np.min(dv)
                if min_dv < 4.99:
                    num_collisions += 1
                    depth = 5.0 - min_dv
                    collision_cost += penalty_base + penalty_slope * depth

        cost = total_len + collision_cost
        return cost, num_collisions

    def _estimate_initial_temperature(self, indices, headings, phi_angles, radii, centers, start_node_idx, samples=50, target_prob=0.8):
        """透過樣本擾動估計初始溫度 (使用完整成本但限制最大溫度以優化避障搜尋)"""
        deltas = []
        n = len(indices)
        current_cost, _ = self._get_combined_cost_and_collisions(indices, headings, phi_angles, radii, centers, start_node_idx)
        
        for _ in range(samples):
            new_idx = list(indices)
            new_h = np.array(headings)
            new_phi = np.array(phi_angles)
            new_r = np.array(radii)
            
            r_val = random.random()
            if r_val < 0.25 and n >= 4:
                idx1, idx2 = random.sample(range(1, n), 2)
                if idx1 > idx2: idx1, idx2 = idx2, idx1
                new_idx = indices[:idx1] + indices[idx1:idx2+1][::-1] + indices[idx2+1:]
            elif r_val < 0.5:
                target = random.randint(0, len(centers) - 1)
                new_h[target] = (new_h[target] + np.random.normal(0, np.radians(15))) % (2 * np.pi)
            elif r_val < 0.75:
                target = random.randint(0, len(centers) - 1)
                if target != start_node_idx:
                    new_phi[target] = (new_phi[target] + np.random.normal(0, np.radians(20))) % (2 * np.pi)
            else:
                target = random.randint(0, len(centers) - 1)
                if target != start_node_idx:
                    new_r[target] = np.clip(new_r[target] + np.random.normal(0, 0.5), self.obs_min_radius, self.obs_max_radius)
            
            new_cost, _ = self._get_combined_cost_and_collisions(new_idx, new_h, new_phi, new_r, centers, start_node_idx)
            deltas.append(abs(new_cost - current_cost))
            
        avg_delta = np.mean(deltas) if deltas else 1.0
        temp = -avg_delta / np.log(target_prob)
        return np.clip(temp, 30.0, 250.0)



    def optimize(self, route, centers, start_node_idx):
        """
        執行 DTSPN 優化。
        :param route: 初始 Route 物件
        :param centers: 目標中心座標 (N, 2)
        :param start_node_idx: 起點索引
        :return: (最佳 Route, 最佳 x-y-heading 點位矩陣, 最佳 phi_angles, 最佳 radii)
        """
        n_points = len(centers)
        current_indices = list(route.indices)
        
        # 確保起點排在第一位
        if current_indices[0] != start_node_idx:
            s_pos = current_indices.index(start_node_idx)
            current_indices = current_indices[s_pos:] + current_indices[:s_pos]
            
        # 1. 智慧初始化變數
        current_headings = np.zeros(n_points)
        current_phi_angles = np.zeros(n_points)
        current_radii = np.ones(n_points) * self.obs_max_radius  # 預設都在最外圈 9km 以利避障
        
        n_route = len(current_indices)
        for pos in range(n_route):
            target_idx = current_indices[pos]
            if target_idx == start_node_idx:
                continue
            prev_idx = current_indices[(pos - 1) % n_route]
            next_idx = current_indices[(pos + 1) % n_route]
            
            # 拜訪點指向前一點與後一點的平分線方向，減少折返
            vec_prev = centers[prev_idx] - centers[target_idx]
            vec_next = centers[next_idx] - centers[target_idx]
            dir_vec = vec_prev / np.linalg.norm(vec_prev) + vec_next / np.linalg.norm(vec_next)
            if np.linalg.norm(dir_vec) < 1e-3:
                dir_vec = vec_next
            
            angle = np.arctan2(dir_vec[1], dir_vec[0])
            current_phi_angles[target_idx] = angle
            
            # 預設航向指向下一個目標的預估拜訪點
            r_next = self.obs_max_radius if next_idx != start_node_idx else 0.0
            next_est_pos = centers[next_idx] + r_next * np.array([np.cos(current_phi_angles[next_idx]), np.sin(current_phi_angles[next_idx])])
            curr_pos = centers[target_idx] + self.obs_max_radius * np.array([np.cos(angle), np.sin(angle)])
            vec_to_next = next_est_pos - curr_pos
            current_headings[target_idx] = np.arctan2(vec_to_next[1], vec_to_next[0])

        # 起點航向指向第二個點
        next_idx = current_indices[1]
        next_est_pos = centers[next_idx] + self.obs_max_radius * np.array([np.cos(current_phi_angles[next_idx]), np.sin(current_phi_angles[next_idx])])
        vec_to_next = next_est_pos - centers[start_node_idx]
        current_headings[start_node_idx] = np.arctan2(vec_to_next[1], vec_to_next[0])

        current_cost, current_col = self._get_combined_cost_and_collisions(
            current_indices, current_headings, current_phi_angles, current_radii, centers, start_node_idx
        )
        
        best_indices = list(current_indices)
        best_headings = np.array(current_headings)
        best_phi_angles = np.array(current_phi_angles)
        best_radii = np.array(current_radii)
        best_cost = current_cost
        best_col = current_col

        # 估計初始溫度
        temp = self._estimate_initial_temperature(
            current_indices, current_headings, current_phi_angles, current_radii, centers, start_node_idx
        )
        temp = max(temp, 100.0)  # 保證有足夠溫度
        print(f"DTSPN 適應性初始溫度: {temp:.4f}，初始碰撞次數: {current_col}")

        # 模擬退火主迴圈
        for step in range(self.max_iter):
            t_curr = temp * (self.cooling_rate ** step)
            
            new_indices = list(current_indices)
            new_headings = np.array(current_headings)
            new_phi_angles = np.array(current_phi_angles)
            new_radii = np.array(current_radii)
            
            r_val = random.random()
            if r_val < 0.15:
                # 1. 2-opt 順序改變 (保持起點在 index 0)
                if n_route >= 4:
                    idx1, idx2 = random.sample(range(1, n_route), 2)
                    if idx1 > idx2: idx1, idx2 = idx2, idx1
                    new_indices = current_indices[:idx1] + current_indices[idx1:idx2+1][::-1] + current_indices[idx2+1:]
            elif r_val < 0.35:
                # 2. 航向角擾動
                target_idx = random.randint(0, n_points - 1)
                new_headings[target_idx] += np.random.normal(0, np.radians(15))
                new_headings[target_idx] %= (2 * np.pi)
            elif r_val < 0.55:
                # 3. 甜甜圈觀測角 phi 擾動
                target_idx = random.randint(0, n_points - 1)
                if target_idx != start_node_idx:
                    new_phi_angles[target_idx] += np.random.normal(0, np.radians(20))
                    new_phi_angles[target_idx] %= (2 * np.pi)
            elif r_val < 0.75:
                # 4. 甜甜圈觀測半徑 r 擾動
                target_idx = random.randint(0, n_points - 1)
                if target_idx != start_node_idx:
                    new_radii[target_idx] += np.random.normal(0, 0.5)
                    new_radii[target_idx] = np.clip(new_radii[target_idx], self.obs_min_radius, self.obs_max_radius)
            else:
                # 5. 智慧航向與角度對齊
                target_pos = random.randint(0, n_route - 1)
                target_node = new_indices[target_pos]
                
                # 計算目前所有拜訪點座標
                pts = np.zeros((n_points, 2))
                for idx in range(n_points):
                    if idx == start_node_idx:
                        pts[idx] = centers[idx]
                    else:
                        pts[idx] = centers[idx] + new_radii[idx] * np.array([np.cos(new_phi_angles[idx]), np.sin(new_phi_angles[idx])])
                
                prev_node = new_indices[(target_pos - 1) % n_route]
                next_node = new_indices[(target_pos + 1) % n_route]
                
                if random.random() < 0.5:
                    # 航向對齊：指向下一點
                    vec = pts[next_node] - pts[target_node]
                    new_headings[target_node] = np.arctan2(vec[1], vec[0])
                else:
                    # 拜訪點角度對齊：指向前後兩點向量的平分線方向
                    if target_node != start_node_idx:
                        vec_prev = pts[prev_node] - centers[target_node]
                        vec_next = pts[next_node] - centers[target_node]
                        dir_vec = vec_prev / np.linalg.norm(vec_prev) + vec_next / np.linalg.norm(vec_next)
                        if np.linalg.norm(dir_vec) > 1e-3:
                            new_phi_angles[target_node] = np.arctan2(dir_vec[1], dir_vec[0])

            new_cost, new_col = self._get_combined_cost_and_collisions(
                new_indices, new_headings, new_phi_angles, new_radii, centers, start_node_idx
            )
            
            delta = new_cost - current_cost
            if delta < 0 or (t_curr > 0 and random.random() < np.exp(-delta / t_curr)):
                current_indices = new_indices
                current_headings = new_headings
                current_phi_angles = new_phi_angles
                current_radii = new_radii
                current_cost = new_cost
                current_col = new_col
                
                if current_cost < best_cost:
                    best_indices = list(current_indices)
                    best_headings = np.array(current_headings)
                    best_phi_angles = np.array(current_phi_angles)
                    best_radii = np.array(current_radii)
                    best_cost = current_cost
                    best_col = current_col

        # 重建最佳航路下的拜訪點座標矩陣 (N, 3: x, y, heading)
        optimized_points = np.zeros((n_points, 3))
        for i in range(n_points):
            if i == start_node_idx:
                optimized_points[i, 0] = centers[i, 0]
                optimized_points[i, 1] = centers[i, 1]
            else:
                optimized_points[i, 0] = centers[i, 0] + best_radii[i] * np.cos(best_phi_angles[i])
                optimized_points[i, 1] = centers[i, 1] + best_radii[i] * np.sin(best_phi_angles[i])
            optimized_points[i, 2] = best_headings[i]
            
        print(f"DTSPN 優化完成。最優成本(含懲罰): {best_cost:.2f}，碰撞次數: {best_col}")
        return Route(best_indices), optimized_points, best_phi_angles, best_radii
