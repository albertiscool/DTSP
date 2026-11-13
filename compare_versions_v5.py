"""
=============================================================================
三版本同環境基準測試 (Three-Version Benchmark & Comparison)
=============================================================================
評估三個演算法版本在完全相同的 25 艘靜態敵艦與 5 艘動態新敵艦環境下的表現：
1. Baseline (學長原版 main): 隨機初始解 + 獨立隨機航向 + 低碰撞懲罰
2. Smart Init & Safe (feature/smart-heading-optimization): 2-opt初解 + 高碰撞懲罰
3. Tangent Smoothing (feature/tangent-heading-smoothing): 2-opt初解 + 全域切線流平滑 + 零打結 + 高碰撞懲罰
=============================================================================
"""

import os
import sys
import time
import random
import numpy as np
import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 將當前資料夾與 dtsp_uav 加入 Python 模組搜尋路徑
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("dtsp_uav"))

from dtsp_uav.core.dubins import DubinsCost
from dtsp_uav.core.routing import Route


# =============================================================================
# 輔助演算法函數
# =============================================================================

def build_2opt_euclidean_tour(centers, start_idx):
    """貪婪最近鄰 + 2-opt 歐氏無交叉初解"""
    unvisited = set(range(len(centers)))
    unvisited.remove(start_idx)
    tour = [start_idx]
    curr = start_idx
    while unvisited:
        next_node = min(unvisited, key=lambda idx: np.linalg.norm(centers[curr] - centers[idx]))
        tour.append(next_node)
        unvisited.remove(next_node)
        curr = next_node
    
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


def insert_cheapest(route_indices, new_idx, centers, do_2opt=False):
    """Cheapest Insertion 動態插入新點"""
    best_pos = None
    best_increase = float('inf')
    n = len(route_indices)
    for i in range(n):
        idx_a = route_indices[i]
        idx_b = route_indices[(i + 1) % n]
        d_a_new = np.linalg.norm(centers[idx_a] - centers[new_idx])
        d_new_b = np.linalg.norm(centers[new_idx] - centers[idx_b])
        d_a_b = np.linalg.norm(centers[idx_a] - centers[idx_b])
        inc = d_a_new + d_new_b - d_a_b
        if inc < best_increase:
            best_increase = inc
            best_pos = i + 1
            
    res = list(route_indices)
    res.insert(best_pos, new_idx)
    
    if do_2opt:
        improved = True
        n = len(res)
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    a, b = res[i - 1], res[i]
                    c, d = res[j], res[(j + 1) % n]
                    d0 = np.linalg.norm(centers[a] - centers[b]) + np.linalg.norm(centers[c] - centers[d])
                    d1 = np.linalg.norm(centers[a] - centers[c]) + np.linalg.norm(centers[b] - centers[d])
                    if d1 < d0 - 1e-4:
                        res[i:j + 1] = res[i:j + 1][::-1]
                        improved = True
                        break
                if improved:
                    break
    return res


# =============================================================================
# 3 種優化器實作
# =============================================================================

class DTSPNSolverBase:
    def __init__(self, turning_radius=2.0, obs_min=5.0, obs_max=9.0):
        self.turning_radius = turning_radius
        self.obs_min_radius = obs_min
        self.obs_max_radius = obs_max
        self.obstacle_radius = obs_min
        self.cost_calculator = DubinsCost(turning_radius)

    def calc_cost_and_collisions(self, indices, points, centers, start_node_idx, penalty_base=50000.0, penalty_slope=50000.0):
        total_len = 0.0
        num_collisions = 0
        collision_cost = 0.0
        n_route = len(indices)
        n_points = len(centers)
        
        for i in range(n_route):
            u = indices[i]
            v = indices[(i + 1) % n_route]
            p1 = points[u]
            p2 = points[v]
            
            t, p, q, mode, length = self.cost_calculator._plan_dubins(p1, p2)
            if mode is None:
                dist = np.linalg.norm(p1[:2] - p2[:2])
                length = dist
                px = np.linspace(p1[0], p2[0], 25)
                py = np.linspace(p1[1], p2[1], 25)
            else:
                px, py = self.cost_calculator._interpolate(p1, t, p, q, mode, step_size=1.0)
                
            total_len += length
            
            # 檢查障礙物
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
                        depth = self.obstacle_radius - np.sqrt(dist_sq)
                        collision_cost += penalty_base + penalty_slope * depth
                        
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
                    
        return total_len, collision_cost, num_collisions


# -----------------------------------------------------------------------------
# 版本 1: 学長 Baseline (main)
# -----------------------------------------------------------------------------
class Version1_Baseline(DTSPNSolverBase):
    def solve(self, centers, start_node_idx, max_iter=2500, cooling_rate=0.996):
        n = len(centers)
        initial_indices = list(range(n))
        initial_indices.remove(start_node_idx)
        indices = [start_node_idx] + initial_indices
        
        headings = np.zeros(n)
        phi_angles = np.zeros(n)
        radii = np.ones(n) * self.obs_max_radius
        
        # 初始角度與航向
        for pos in range(n):
            target_idx = indices[pos]
            if target_idx == start_node_idx: continue
            prev_idx = indices[(pos - 1) % n]
            next_idx = indices[(pos + 1) % n]
            vec_prev = centers[prev_idx] - centers[target_idx]
            vec_next = centers[next_idx] - centers[target_idx]
            dir_vec = vec_prev / np.linalg.norm(vec_prev) + vec_next / np.linalg.norm(vec_next)
            if np.linalg.norm(dir_vec) < 1e-3: dir_vec = vec_next
            phi_angles[target_idx] = np.arctan2(dir_vec[1], dir_vec[0])
            
            curr_pos = centers[target_idx] + self.obs_max_radius * np.array([np.cos(phi_angles[target_idx]), np.sin(phi_angles[target_idx])])
            r_next = self.obs_max_radius if next_idx != start_node_idx else 0.0
            next_pos = centers[next_idx] + r_next * np.array([np.cos(phi_angles[next_idx]), np.sin(phi_angles[next_idx])])
            vec = next_pos - curr_pos
            headings[target_idx] = np.arctan2(vec[1], vec[0])
            
        def get_points(h, p, r):
            pts = np.zeros((n, 3))
            for i in range(n):
                if i == start_node_idx:
                    pts[i, :2] = centers[i]
                else:
                    pts[i, 0] = centers[i, 0] + r[i] * np.cos(p[i])
                    pts[i, 1] = centers[i, 1] + r[i] * np.sin(p[i])
                pts[i, 2] = h[i]
            return pts
            
        curr_pts = get_points(headings, phi_angles, radii)
        t_len, c_cost, c_col = self.calc_cost_and_collisions(indices, curr_pts, centers, start_node_idx, penalty_base=2000.0, penalty_slope=3000.0)
        curr_cost = t_len + c_cost
        
        best_indices = list(indices)
        best_h, best_p, best_r = np.array(headings), np.array(phi_angles), np.array(radii)
        best_cost = curr_cost
        best_len, best_col = t_len, c_col
        
        temp = 200.0
        for step in range(max_iter):
            t_curr = temp * (cooling_rate ** step)
            new_idx = list(indices)
            new_h, new_p, new_r = np.array(headings), np.array(phi_angles), np.array(radii)
            
            r_val = random.random()
            if r_val < 0.20 and n >= 4:
                idx1, idx2 = random.sample(range(1, n), 2)
                if idx1 > idx2: idx1, idx2 = idx2, idx1
                new_idx = indices[:idx1] + indices[idx1:idx2+1][::-1] + indices[idx2+1:]
            elif r_val < 0.50:
                t_i = random.randint(0, n - 1)
                new_h[t_i] = (new_h[t_i] + np.random.normal(0, np.radians(15))) % (2 * np.pi)
            elif r_val < 0.75:
                t_i = random.randint(0, n - 1)
                if t_i != start_node_idx:
                    new_p[t_i] = (new_p[t_i] + np.random.normal(0, np.radians(20))) % (2 * np.pi)
            else:
                t_i = random.randint(0, n - 1)
                if t_i != start_node_idx:
                    new_r[t_i] = np.clip(new_r[t_i] + np.random.normal(0, 0.5), self.obs_min_radius, self.obs_max_radius)
                    
            trial_pts = get_points(new_h, new_p, new_r)
            t_len, c_cost, c_col = self.calc_cost_and_collisions(new_idx, trial_pts, centers, start_node_idx, penalty_base=2000.0, penalty_slope=3000.0)
            trial_cost = t_len + c_cost
            
            delta = trial_cost - curr_cost
            if delta < 0 or (t_curr > 0 and random.random() < np.exp(-delta / t_curr)):
                indices, headings, phi_angles, radii = new_idx, new_h, new_p, new_r
                curr_cost = trial_cost
                if curr_cost < best_cost:
                    best_indices = list(indices)
                    best_h, best_p, best_r = np.array(headings), np.array(phi_angles), np.array(radii)
                    best_cost = curr_cost
                    best_len, best_col = t_len, c_col
                    
        final_pts = get_points(best_h, best_p, best_r)
        # 精確評估實際總長與碰撞
        final_len, _, final_col = self.calc_cost_and_collisions(best_indices, final_pts, centers, start_node_idx)
        return best_indices, final_pts, final_len, final_col


# -----------------------------------------------------------------------------
# 版本 2: Smart Init & Safe (feature/smart-heading-optimization)
# -----------------------------------------------------------------------------
class Version2_SmartInit(DTSPNSolverBase):
    def solve(self, centers, start_node_idx, max_iter=2500, cooling_rate=0.996):
        n = len(centers)
        indices = build_2opt_euclidean_tour(centers, start_node_idx)
        
        headings = np.zeros(n)
        phi_angles = np.zeros(n)
        radii = np.ones(n) * self.obs_max_radius
        
        for pos in range(n):
            target_idx = indices[pos]
            if target_idx == start_node_idx: continue
            prev_idx = indices[(pos - 1) % n]
            next_idx = indices[(pos + 1) % n]
            vec_prev = centers[prev_idx] - centers[target_idx]
            vec_next = centers[next_idx] - centers[target_idx]
            dir_vec = vec_prev / np.linalg.norm(vec_prev) + vec_next / np.linalg.norm(vec_next)
            if np.linalg.norm(dir_vec) < 1e-3: dir_vec = vec_next
            phi_angles[target_idx] = np.arctan2(dir_vec[1], dir_vec[0])
            
            curr_pos = centers[target_idx] + self.obs_max_radius * np.array([np.cos(phi_angles[target_idx]), np.sin(phi_angles[target_idx])])
            r_next = self.obs_max_radius if next_idx != start_node_idx else 0.0
            next_pos = centers[next_idx] + r_next * np.array([np.cos(phi_angles[next_idx]), np.sin(phi_angles[next_idx])])
            vec = next_pos - curr_pos
            headings[target_idx] = np.arctan2(vec[1], vec[0])
            
        def get_points(h, p, r):
            pts = np.zeros((n, 3))
            for i in range(n):
                if i == start_node_idx:
                    pts[i, :2] = centers[i]
                else:
                    pts[i, 0] = centers[i, 0] + r[i] * np.cos(p[i])
                    pts[i, 1] = centers[i, 1] + r[i] * np.sin(p[i])
                pts[i, 2] = h[i]
            return pts
            
        curr_pts = get_points(headings, phi_angles, radii)
        t_len, c_cost, c_col = self.calc_cost_and_collisions(indices, curr_pts, centers, start_node_idx, penalty_base=50000.0, penalty_slope=50000.0)
        curr_cost = t_len + c_cost
        
        best_indices = list(indices)
        best_h, best_p, best_r = np.array(headings), np.array(phi_angles), np.array(radii)
        best_cost = curr_cost
        best_len, best_col = t_len, c_col
        
        temp = 200.0
        for step in range(max_iter):
            t_curr = temp * (cooling_rate ** step)
            new_idx = list(indices)
            new_h, new_p, new_r = np.array(headings), np.array(phi_angles), np.array(radii)
            
            r_val = random.random()
            if r_val < 0.20 and n >= 4:
                idx1, idx2 = random.sample(range(1, n), 2)
                if idx1 > idx2: idx1, idx2 = idx2, idx1
                new_idx = indices[:idx1] + indices[idx1:idx2+1][::-1] + indices[idx2+1:]
            elif r_val < 0.45:
                t_i = random.randint(0, n - 1)
                new_h[t_i] = (new_h[t_i] + np.random.normal(0, np.radians(15))) % (2 * np.pi)
            elif r_val < 0.70:
                t_i = random.randint(0, n - 1)
                if t_i != start_node_idx:
                    new_p[t_i] = (new_p[t_i] + np.random.normal(0, np.radians(20))) % (2 * np.pi)
            else:
                t_i = random.randint(0, n - 1)
                if t_i != start_node_idx:
                    new_r[t_i] = np.clip(new_r[t_i] + np.random.normal(0, 0.5), self.obs_min_radius, self.obs_max_radius)
                    
            trial_pts = get_points(new_h, new_p, new_r)
            t_len, c_cost, c_col = self.calc_cost_and_collisions(new_idx, trial_pts, centers, start_node_idx, penalty_base=50000.0, penalty_slope=50000.0)
            trial_cost = t_len + c_cost
            
            delta = trial_cost - curr_cost
            if delta < 0 or (t_curr > 0 and random.random() < np.exp(-delta / t_curr)):
                indices, headings, phi_angles, radii = new_idx, new_h, new_p, new_r
                curr_cost = trial_cost
                if curr_cost < best_cost:
                    best_indices = list(indices)
                    best_h, best_p, best_r = np.array(headings), np.array(phi_angles), np.array(radii)
                    best_cost = curr_cost
                    best_len, best_col = t_len, c_col
                    
        final_pts = get_points(best_h, best_p, best_r)
        final_len, _, final_col = self.calc_cost_and_collisions(best_indices, final_pts, centers, start_node_idx)
        return best_indices, final_pts, final_len, final_col


# -----------------------------------------------------------------------------
# 版本 3: Tangent Heading Smoothing (feature/tangent-heading-smoothing)
# -----------------------------------------------------------------------------
class Version3_TangentSmoothing(DTSPNSolverBase):
    def compute_tangents(self, indices, phi_angles, radii, centers, start_node_idx):
        n = len(centers)
        pts = np.zeros((n, 2))
        for idx in range(n):
            if idx == start_node_idx:
                pts[idx] = centers[idx]
            else:
                pts[idx] = centers[idx] + radii[idx] * np.array([np.cos(phi_angles[idx]), np.sin(phi_angles[idx])])
        headings = np.zeros(n)
        for pos in range(n):
            curr_node = indices[pos]
            prev_node = indices[(pos - 1) % n]
            next_node = indices[(pos + 1) % n]
            v_in = pts[curr_node] - pts[prev_node]
            n_in = np.linalg.norm(v_in)
            if n_in > 1e-6: v_in /= n_in
            v_out = pts[next_node] - pts[curr_node]
            n_out = np.linalg.norm(v_out)
            if n_out > 1e-6: v_out /= n_out
            v_t = v_in + v_out
            if np.linalg.norm(v_t) < 1e-3: v_t = v_out
            else: v_t /= np.linalg.norm(v_t)
            headings[curr_node] = np.arctan2(v_t[1], v_t[0])
        return headings

    def solve(self, centers, start_node_idx, max_iter=2500, cooling_rate=0.996):
        n = len(centers)
        indices = build_2opt_euclidean_tour(centers, start_node_idx)
        
        phi_angles = np.zeros(n)
        radii = np.ones(n) * self.obs_max_radius
        centroid = np.mean(centers, axis=0)
        
        for pos in range(n):
            target_idx = indices[pos]
            if target_idx == start_node_idx: continue
            prev_idx = indices[(pos - 1) % n]
            next_idx = indices[(pos + 1) % n]
            vec_to_c = centers[target_idx] - centroid
            vec_prev = centers[target_idx] - centers[prev_idx]
            vec_next = centers[next_idx] - centers[target_idx]
            tangent = vec_next / (np.linalg.norm(vec_next) + 1e-6) + vec_prev / (np.linalg.norm(vec_prev) + 1e-6)
            if np.linalg.norm(tangent) < 1e-3:
                out_n = vec_to_c
            else:
                out_n = np.array([-tangent[1], tangent[0]])
                if np.dot(out_n, vec_to_c) < 0: out_n = -out_n
            phi_angles[target_idx] = np.arctan2(out_n[1], out_n[0]) if np.linalg.norm(out_n) > 1e-3 else np.arctan2(vec_to_c[1], vec_to_c[0])
            
        headings = self.compute_tangents(indices, phi_angles, radii, centers, start_node_idx)
        
        def get_points(h, p, r):
            pts = np.zeros((n, 3))
            for i in range(n):
                if i == start_node_idx:
                    pts[i, :2] = centers[i]
                else:
                    pts[i, 0] = centers[i, 0] + r[i] * np.cos(p[i])
                    pts[i, 1] = centers[i, 1] + r[i] * np.sin(p[i])
                pts[i, 2] = h[i]
            return pts
            
        curr_pts = get_points(headings, phi_angles, radii)
        t_len, c_cost, c_col = self.calc_cost_and_collisions(indices, curr_pts, centers, start_node_idx, penalty_base=50000.0, penalty_slope=50000.0)
        curr_cost = t_len + c_cost
        
        best_indices = list(indices)
        best_h, best_p, best_r = np.array(headings), np.array(phi_angles), np.array(radii)
        best_cost = curr_cost
        best_len, best_col = t_len, c_col
        
        temp = 200.0
        for step in range(max_iter):
            t_curr = temp * (cooling_rate ** step)
            new_idx = list(indices)
            new_p, new_r = np.array(phi_angles), np.array(radii)
            
            r_val = random.random()
            if r_val < 0.20 and n >= 4:
                idx1, idx2 = random.sample(range(1, n), 2)
                if idx1 > idx2: idx1, idx2 = idx2, idx1
                new_idx = indices[:idx1] + indices[idx1:idx2+1][::-1] + indices[idx2+1:]
            elif r_val < 0.65:
                t_i = random.randint(0, n - 1)
                if t_i != start_node_idx:
                    new_p[t_i] = (new_p[t_i] + np.random.normal(0, np.radians(15))) % (2 * np.pi)
            else:
                t_i = random.randint(0, n - 1)
                if t_i != start_node_idx:
                    new_r[t_i] = np.clip(new_r[t_i] + np.random.normal(0, 0.4), self.obs_min_radius, self.obs_max_radius)
                    
            new_h = self.compute_tangents(new_idx, new_p, new_r, centers, start_node_idx)
            trial_pts = get_points(new_h, new_p, new_r)
            t_len, c_cost, c_col = self.calc_cost_and_collisions(new_idx, trial_pts, centers, start_node_idx, penalty_base=50000.0, penalty_slope=50000.0)
            trial_cost = t_len + c_cost
            
            delta = trial_cost - curr_cost
            if delta < 0 or (t_curr > 0 and random.random() < np.exp(-delta / t_curr)):
                indices, headings, phi_angles, radii = new_idx, new_h, new_p, new_r
                curr_cost = trial_cost
                if curr_cost < best_cost:
                    best_indices = list(indices)
                    best_h, best_p, best_r = np.array(headings), np.array(phi_angles), np.array(radii)
                    best_cost = curr_cost
                    best_len, best_col = t_len, c_col
                    
        # 局部切線精細化
        for _ in range(2):
            for t_i in range(n):
                if t_i == start_node_idx: continue
                orig_p = best_p[t_i]
                for d_phi in [-0.1, 0.1]:
                    test_p = np.array(best_p)
                    test_p[t_i] = (orig_p + d_phi) % (2 * np.pi)
                    test_h = self.compute_tangents(best_indices, test_p, best_r, centers, start_node_idx)
                    test_pts = get_points(test_h, test_p, best_r)
                    t_len, c_cost, c_col = self.calc_cost_and_collisions(best_indices, test_pts, centers, start_node_idx, penalty_base=50000.0, penalty_slope=50000.0)
                    if t_len + c_cost < best_cost:
                        best_cost = t_len + c_cost
                        best_p = test_p
                        best_h = test_h
                        
        final_pts = get_points(best_h, best_p, best_r)
        final_len, _, final_col = self.calc_cost_and_collisions(best_indices, final_pts, centers, start_node_idx)
        return best_indices, final_pts, final_len, final_col



# -----------------------------------------------------------------------------
# 版本 5: Transformer V5 + 2-Opt + True Physical 0-Collision Optimizer
# -----------------------------------------------------------------------------
class Version5_ZeroCollisionFast(Version3_TangentSmoothing):
    def __init__(self, model_path="transformer_checkpoints/best_model.pt", turning_radius=2.0, obs_min=6.2, obs_max=9.0):
        super().__init__(turning_radius, obs_min, obs_max)
        self.obstacle_radius = 5.15
        # ``obs_min`` is the normal SA operating bound inherited from V5.  It
        # must not be confused with the physical observation constraint: an
        # escape search is allowed to use the complete [5, 9] km annulus.
        self.physical_obs_min_radius = 5.0
        self.physical_obs_max_radius = 9.0
        self.wall_penalty = 200000.0
        self.corridor_penalty = 2500.0
        self.last_topology_diagnostics = {}
        self.last_adaptive_radius_repairs = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model definition
        sys.path.append(os.path.abspath("."))
        try:
            from train_transformer_dtsp import DTSPTransformerPolicy, EMBED_DIM, N_HEADS, N_ENC_LAYERS, FF_DIM, FILTER_K, DROPOUT
            self.policy = DTSPTransformerPolicy(
                embed_dim=EMBED_DIM, n_heads=N_HEADS, n_enc_layers=N_ENC_LAYERS, 
                ff_dim=FF_DIM, filter_k=FILTER_K, dropout=DROPOUT
            ).to(self.device)
            
            if os.path.exists(model_path):
                ckpt = torch.load(model_path, map_location=self.device)
                self.policy.load_state_dict(ckpt['policy'])
                self.policy.eval()
            else:
                print(f"[WARN] V5 Model not found at {model_path}!")
        except Exception as e:
            print(f"[ERROR] Failed to load V5 Model: {e}")
            self.policy = None

    def calc_cost_and_collisions(self, indices, points, centers, start_node_idx, penalty_base=200000.0, penalty_slope=200000.0):
        total_len = 0.0
        num_collisions = 0
        collision_cost = 0.0
        n_route = len(indices)
        n_points = len(centers)
        safe_r = self.obstacle_radius
        
        for i in range(n_route):
            u = indices[i]
            v = indices[(i + 1) % n_route]
            p1 = points[u]
            p2 = points[v]
            
            t, p, q, mode, length = self.cost_calculator._plan_dubins(p1, p2)
            if mode is None:
                px = np.linspace(p1[0], p2[0], 25)
                py = np.linspace(p1[1], p2[1], 25)
                length = np.linalg.norm(p1[:2] - p2[:2])
            else:
                px, py = self.cost_calculator._interpolate(p1, t, p, q, mode, step_size=0.6)
            total_len += length
            
            for o_idx in range(n_points):
                if o_idx == start_node_idx: continue
                d = np.sqrt((px - centers[o_idx, 0])**2 + (py - centers[o_idx, 1])**2)
                min_d = np.min(d)
                if min_d < safe_r:
                    num_collisions += 1
                    collision_cost += penalty_base + penalty_slope * (safe_r - min_d)
                    
        return total_len, collision_cost, num_collisions

    def check_segment_collision(self, p1, p2, centers, start_node_idx, safe_r=5.05):
        t, p, q, mode, length = self.cost_calculator._plan_dubins(p1, p2)
        if mode is None:
            px = np.linspace(p1[0], p2[0], 30)
            py = np.linspace(p1[1], p2[1], 30)
            length = np.linalg.norm(p1[:2] - p2[:2])
        else:
            px, py = self.cost_calculator._interpolate(p1, t, p, q, mode, step_size=0.15)
        for o_idx, c in enumerate(centers):
            if o_idx == start_node_idx: continue
            d = np.sqrt((px - c[0])**2 + (py - c[1])**2)
            if np.min(d) < safe_r:
                return True, length, np.min(d)
        return False, length, np.min(d)

    def _normalise_tour(self, tour, n, start_node_idx):
        """Validate a decoder tour and rotate it so that the depot is fixed."""
        tour = [int(node) for node in tour]
        if len(tour) != n or set(tour) != set(range(n)):
            return build_2opt_euclidean_tour(np.asarray(tour), 0) if n == 0 else None
        start_pos = tour.index(start_node_idx)
        return tour[start_pos:] + tour[:start_pos]

    def infer_transformer_tour(self, centers, start_node_idx):
        """Run the policy on the *current complete* target set.

        This is deliberately a full inference rather than an insertion
        heuristic.  The Transformer only receives coordinates and observation
        radii, so supplying every currently-known target is how a dynamic map
        update is fed back into the learned topology policy.
        """
        n = len(centers)
        if self.policy is None:
            return build_2opt_euclidean_tour(centers, start_node_idx)

        coords = np.zeros((n, 2), dtype=np.float32)
        coords[:, 0] = (centers[:, 0] - 50.0) / 100.0
        coords[:, 1] = (centers[:, 1] - 0.0) / 100.0
        obs_radii = np.ones(n, dtype=np.float32) * (self.obs_max_radius / 100.0)
        obs_radii[start_node_idx] = 0.0
        coords_t = torch.tensor(coords, device=self.device).unsqueeze(0)
        obs_radii_t = torch.tensor(obs_radii, device=self.device).unsqueeze(0)
        with torch.no_grad():
            tours, _ = self.policy(coords_t, obs_radii_t, greedy=True, start_city=start_node_idx)
        tour = self._normalise_tour(tours[0].cpu().numpy().tolist(), n, start_node_idx)
        return tour if tour is not None else build_2opt_euclidean_tour(centers, start_node_idx)

    def replan_dynamic(self, existing_centers, new_targets, start_node_idx, max_iter=600, cooling_rate=0.995):
        """Replan after a dynamic arrival by re-inferring the global topology.

        ``new_targets`` may be one target or a batch.  No target is cheapest-
        inserted into the old sequence; the updated target set is sent through
        the Transformer once and the resulting topology is then repaired and
        physically optimized by the normal V5 pipeline.
        """
        existing = np.asarray(existing_centers, dtype=float)
        additions = np.asarray(new_targets, dtype=float)
        if additions.ndim == 1:
            additions = additions.reshape(1, -1)
        if existing.ndim != 2 or existing.shape[1] != 2 or additions.ndim != 2 or additions.shape[1] != 2:
            raise ValueError("existing_centers and new_targets must have shape (N, 2)")
        updated_centers = np.vstack([existing, additions])
        transformer_tour = self.infer_transformer_tour(updated_centers, start_node_idx)
        return self.solve(
            updated_centers,
            start_node_idx,
            max_iter=max_iter,
            cooling_rate=cooling_rate,
            initial_indices=transformer_tour,
        )

    @staticmethod
    def _point_to_segment_distance(point, p1, p2):
        segment = p2 - p1
        norm_sq = float(np.dot(segment, segment))
        if norm_sq <= 1e-12:
            return float(np.linalg.norm(point - p1)), 0.0
        projection = float(np.clip(np.dot(point - p1, segment) / norm_sq, 0.0, 1.0))
        return float(np.linalg.norm(point - (p1 + projection * segment))), projection

    def _edge_occlusion(self, centers, u, v, start_node_idx, corridor_radius=6.5):
        """Return corridor blockers and connected 3+-obstacle wall clusters."""
        blockers = []
        p1, p2 = centers[u], centers[v]
        for obstacle_idx, center in enumerate(centers):
            if obstacle_idx in (u, v, start_node_idx):
                continue
            distance, projection = self._point_to_segment_distance(center, p1, p2)
            if distance < corridor_radius:
                blockers.append((obstacle_idx, projection, distance))

        # A solid local wall is a connected chain of overlapping red zones,
        # not simply several unrelated targets near the same edge.
        parent = list(range(len(blockers)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        overlap_distance = 2.0 * self.physical_obs_min_radius
        for left in range(len(blockers)):
            for right in range(left + 1, len(blockers)):
                left_idx, right_idx = blockers[left][0], blockers[right][0]
                if np.linalg.norm(centers[left_idx] - centers[right_idx]) <= overlap_distance + 1e-9:
                    union(left, right)

        clusters = {}
        for blocker_pos, blocker in enumerate(blockers):
            clusters.setdefault(find(blocker_pos), []).append(blocker)
        solid_walls = [cluster for cluster in clusters.values() if len(cluster) >= 3]
        return blockers, solid_walls

    def detect_impassable_walls(self, tour, centers, start_node_idx, corridor_radius=6.5):
        """Locate sequence edges occluded by a chain of 3+ overlapping zones."""
        walls = []
        for position, u in enumerate(tour):
            v = tour[(position + 1) % len(tour)]
            blockers, clusters = self._edge_occlusion(
                centers, u, v, start_node_idx, corridor_radius=corridor_radius
            )
            for cluster in clusters:
                walls.append({
                    "position": position,
                    "edge": (u, v),
                    "obstacles": [item[0] for item in cluster],
                    "blocker_count": len(blockers),
                })
        return walls

    def _topology_cost_matrix(self, centers, start_node_idx, corridor_radius=6.5):
        n = len(centers)
        cost_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d_euc = np.linalg.norm(centers[i] - centers[j])
                blockers, walls = self._edge_occlusion(
                    centers, i, j, start_node_idx, corridor_radius=corridor_radius
                )
                cost_matrix[i, j] = d_euc + self.corridor_penalty * len(blockers)
                if walls:
                    cost_matrix[i, j] += self.wall_penalty * len(walls)
        return cost_matrix

    @staticmethod
    def _tour_cost(tour, cost_matrix):
        return sum(cost_matrix[tour[pos], tour[(pos + 1) % len(tour)]] for pos in range(len(tour)))

    @staticmethod
    def _or_opt_move(tour, start, length, insert_at):
        """Move a contiguous block without allowing the depot to move."""
        block = tour[start:start + length]
        if not block or 0 in block:
            return None
        remaining = tour[:start] + tour[start + length:]
        if insert_at < 1 or insert_at > len(remaining):
            return None
        return remaining[:insert_at] + block + remaining[insert_at:]

    @staticmethod
    def _three_opt_variants(tour, first_cut, second_cut, third_cut):
        prefix = tour[:first_cut]
        first = tour[first_cut:second_cut]
        second = tour[second_cut:third_cut]
        suffix = tour[third_cut:]
        variants = [
            prefix + first[::-1] + second + suffix,
            prefix + first + second[::-1] + suffix,
            prefix + first[::-1] + second[::-1] + suffix,
            prefix + second + first + suffix,
            prefix + second[::-1] + first + suffix,
            prefix + second + first[::-1] + suffix,
            prefix + second[::-1] + first[::-1] + suffix,
        ]
        seen = set()
        unique = []
        for candidate in variants:
            key = tuple(candidate)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        return unique

    def deep_sequence_repair(self, tour, centers, start_node_idx, corridor_radius=6.5, max_rounds=3):
        """Use Or-Opt and 3-Opt only when a solid-wall edge is detected."""
        tour = list(tour)
        initial_walls = self.detect_impassable_walls(tour, centers, start_node_idx, corridor_radius)
        if not initial_walls:
            return tour, initial_walls, initial_walls

        cost_matrix = self._topology_cost_matrix(centers, start_node_idx, corridor_radius)
        current_score = self._tour_cost(tour, cost_matrix)
        for _ in range(max_rounds):
            walls = self.detect_impassable_walls(tour, centers, start_node_idx, corridor_radius)
            if not walls:
                break
            best_candidate = None
            best_score = current_score

            # First, move 1--3 targets adjacent to an occluded edge.  This is
            # much deeper than a reversal but still concentrated on the cause.
            for wall in walls:
                pos = wall["position"]
                focus = {max(1, min(len(tour) - 1, pos + delta)) for delta in (-1, 0, 1)}
                for block_start in focus:
                    for block_length in range(1, min(3, len(tour) - block_start) + 1):
                        for insert_at in range(1, len(tour) - block_length + 1):
                            candidate = self._or_opt_move(tour, block_start, block_length, insert_at)
                            if candidate is None or candidate == tour:
                                continue
                            score = self._tour_cost(candidate, cost_matrix)
                            if score < best_score - 1e-6:
                                best_candidate, best_score = candidate, score

            # If no relocation escapes the wall, reconnect the two route
            # fragments through bounded 3-Opt alternatives around that edge.
            if best_candidate is None:
                for wall in walls:
                    pos = wall["position"]
                    first_cuts = {
                        max(1, min(len(tour) - 2, pos + delta))
                        for delta in (-1, 0, 1)
                    }
                    for first_cut in first_cuts:
                        for second_cut in range(first_cut + 1, len(tour) - 1):
                            for third_cut in range(second_cut + 1, len(tour) + 1):
                                for candidate in self._three_opt_variants(
                                    tour, first_cut, second_cut, third_cut
                                ):
                                    score = self._tour_cost(candidate, cost_matrix)
                                    if score < best_score - 1e-6:
                                        best_candidate, best_score = candidate, score

            if best_candidate is None:
                break
            tour, current_score = best_candidate, best_score

        final_walls = self.detect_impassable_walls(tour, centers, start_node_idx, corridor_radius)
        return tour, initial_walls, final_walls

    def build_obstacle_aware_tour(self, init_indices, centers, start_node_idx):
        n = len(centers)
        cost_matrix = self._topology_cost_matrix(centers, start_node_idx)

        tour = list(init_indices)
        improved = True
        passes = 0
        while improved and passes < 2 * n:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    new_t = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                    if self._tour_cost(new_t, cost_matrix) < self._tour_cost(tour, cost_matrix) - 1e-4:
                        tour = new_t
                        improved = True
                        passes += 1
                        break
                if improved:
                    break
        repaired_tour, initial_walls, final_walls = self.deep_sequence_repair(
            tour, centers, start_node_idx
        )
        self.last_topology_diagnostics = {
            "initial_wall_count": len(initial_walls),
            "remaining_wall_count": len(final_walls),
            "deep_sequence_repair_used": bool(initial_walls),
        }
        return repaired_tour

    def _adaptive_radius_candidates(self, current_radius):
        """Boundary-aware candidates covering the physical, not SA, annulus."""
        lower = self.physical_obs_min_radius + 0.001
        upper = self.physical_obs_max_radius - 0.001
        anchors = [
            lower, 5.01, 5.05, 5.15, 5.35, 5.75, 6.25, 6.8,
            7.5, 8.2, 8.65, 8.9, 8.99, upper,
        ]
        local = [current_radius + delta for delta in (-1.0, -0.45, -0.15, 0.15, 0.45, 1.0)]
        uniform = np.linspace(lower, upper, 11).tolist()
        return sorted({round(float(np.clip(radius, lower, upper)), 3) for radius in anchors + local + uniform})

    def _route_length_and_collision_count(self, indices, points, centers, start_node_idx, safe_r=5.0):
        total_length = 0.0
        collisions = 0
        for pos, u in enumerate(indices):
            v = indices[(pos + 1) % len(indices)]
            hit, length, _ = self.check_segment_collision(points[u], points[v], centers, start_node_idx, safe_r)
            total_length += length
            collisions += int(hit)
        return total_length, collisions

    def adaptive_radius_repair(self, indices, points, centers, start_node_idx, max_passes=2):
        """Exhaustively probe [5.001, 8.999] only after the normal search fails."""
        repairs = []
        verification_radius = self.physical_obs_min_radius
        for _ in range(max_passes):
            made_repair = False
            for pos, u in enumerate(indices):
                v = indices[(pos + 1) % len(indices)]
                hit, _, _ = self.check_segment_collision(
                    points[u], points[v], centers, start_node_idx, safe_r=verification_radius + 0.001
                )
                if not hit:
                    continue
                for fix_target in (v, u):
                    if fix_target == start_node_idx:
                        continue
                    target_pos = indices.index(fix_target)
                    prev_node = indices[(target_pos - 1) % len(indices)]
                    next_node = indices[(target_pos + 1) % len(indices)]
                    center = centers[fix_target]
                    current_radius = float(np.linalg.norm(points[fix_target, :2] - center))
                    best_candidate = None
                    best_length = float("inf")
                    for radius in self._adaptive_radius_candidates(current_radius):
                        for phi in np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False):
                            candidate_xy = center + radius * np.array([np.cos(phi), np.sin(phi)])
                            if any(
                                np.linalg.norm(candidate_xy - centers[other]) < verification_radius + 0.001
                                for other in range(len(centers))
                                if other not in (fix_target, start_node_idx)
                            ):
                                continue
                            incoming = candidate_xy - points[prev_node, :2]
                            outgoing = points[next_node, :2] - candidate_xy
                            incoming /= np.linalg.norm(incoming) + 1e-9
                            outgoing /= np.linalg.norm(outgoing) + 1e-9
                            average = incoming + outgoing
                            average /= np.linalg.norm(average) + 1e-9
                            base_headings = [
                                points[fix_target, 2],
                                np.arctan2(incoming[1], incoming[0]),
                                np.arctan2(outgoing[1], outgoing[0]),
                                np.arctan2(average[1], average[0]),
                                phi + np.pi / 2,
                                phi - np.pi / 2,
                            ]
                            for base_heading in base_headings:
                                for heading_offset in (0.0, -0.18, 0.18):
                                    candidate = np.array([candidate_xy[0], candidate_xy[1], base_heading + heading_offset])
                                    hit_in, length_in, _ = self.check_segment_collision(
                                        points[prev_node], candidate, centers, start_node_idx, safe_r=verification_radius
                                    )
                                    if hit_in:
                                        continue
                                    hit_out, length_out, _ = self.check_segment_collision(
                                        candidate, points[next_node], centers, start_node_idx, safe_r=verification_radius
                                    )
                                    if hit_out:
                                        continue
                                    if length_in + length_out < best_length:
                                        best_candidate = candidate
                                        best_length = length_in + length_out
                                        best_radius = radius
                    if best_candidate is not None:
                        points[fix_target] = best_candidate
                        repairs.append({"node": int(fix_target), "radius": float(best_radius)})
                        made_repair = True
                        break
            if not made_repair:
                break
        self.last_adaptive_radius_repairs = repairs
        return repairs
        return tour

    def solve(self, centers, start_node_idx, max_iter=600, cooling_rate=0.995, initial_indices=None):
        n = len(centers)
        
        # 1. Transformer Policy Tour or Initial Indices
        if initial_indices is not None:
            indices = list(initial_indices)
        else:
            coords = np.zeros((n, 2), dtype=np.float32)
            coords[:, 0] = (centers[:, 0] - 50.0) / 100.0
            coords[:, 1] = (centers[:, 1] - 0.0) / 100.0
            obs_radii = np.ones(n, dtype=np.float32) * (self.obs_max_radius / 100.0)
            obs_radii[start_node_idx] = 0.0
            coords_t = torch.tensor(coords, device=self.device).unsqueeze(0)
            obs_radii_t = torch.tensor(obs_radii, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                tours, _ = self.policy(coords_t, obs_radii_t, greedy=True, start_city=start_node_idx)
                raw_indices = tours[0].cpu().numpy().tolist()
                
            # 2. Obstacle-Aware 2-Opt Untangling
            indices = self.build_obstacle_aware_tour(raw_indices, centers, start_node_idx)
                
        # 3. Cluster-Aware Outward Initial Placement
        phi_angles = np.zeros(n)
        radii = np.ones(n) * 7.5
        centroid = np.mean(centers, axis=0)
        
        for pos in range(n):
            target_idx = indices[pos]
            if target_idx == start_node_idx: continue
            close_nbs = [j for j in range(n) if j != target_idx and j != start_node_idx and np.linalg.norm(centers[target_idx] - centers[j]) < 7.5]
            if close_nbs:
                vec_away = np.zeros(2)
                for nb in close_nbs:
                    diff = centers[target_idx] - centers[nb]
                    vec_away += diff / (np.linalg.norm(diff) + 1e-6)
                phi_angles[target_idx] = np.arctan2(vec_away[1], vec_away[0])
                radii[target_idx] = 8.0
            else:
                prev_idx = indices[(pos - 1) % n]
                next_idx = indices[(pos + 1) % n]
                vec_prev = centers[target_idx] - centers[prev_idx]
                vec_next = centers[next_idx] - centers[target_idx]
                tangent = vec_next / (np.linalg.norm(vec_next) + 1e-6) + vec_prev / (np.linalg.norm(vec_prev) + 1e-6)
                if np.linalg.norm(tangent) < 1e-3: out_n = centers[target_idx] - centroid
                else:
                    out_n = np.array([-tangent[1], tangent[0]])
                    if np.dot(out_n, centers[target_idx] - centroid) < 0: out_n = -out_n
                phi_angles[target_idx] = np.arctan2(out_n[1], out_n[0])
                radii[target_idx] = 7.5
            
        headings = self.compute_tangents(indices, phi_angles, radii, centers, start_node_idx)
        
        def get_points(h, p, r):
            pts = np.zeros((n, 3))
            for i in range(n):
                if i == start_node_idx: pts[i, :2] = centers[i]
                else:
                    pts[i, 0] = centers[i, 0] + r[i] * np.cos(p[i])
                    pts[i, 1] = centers[i, 1] + r[i] * np.sin(p[i])
                pts[i, 2] = h[i]
            return pts
            
        curr_pts = get_points(headings, phi_angles, radii)
        t_len, c_cost, c_col = self.calc_cost_and_collisions(indices, curr_pts, centers, start_node_idx, penalty_base=500000.0, penalty_slope=500000.0)
        curr_cost = t_len + c_cost
        best_indices = list(indices)
        best_h, best_p, best_r = np.array(headings), np.array(phi_angles), np.array(radii)
        best_cost = curr_cost
        best_col = c_col
        
        # 4. Strict Collision-First SA
        temp = 100.0
        for step in range(max_iter):
            t_curr = temp * (cooling_rate ** step)
            new_p, new_r = np.array(best_p), np.array(best_r)
            
            t_i = random.randint(0, n - 1)
            if t_i != start_node_idx:
                if random.random() < 0.70:
                    new_p[t_i] = (new_p[t_i] + np.random.normal(0, np.radians(20))) % (2 * np.pi)
                else:
                    new_r[t_i] = np.clip(new_r[t_i] + np.random.normal(0, 0.4), 6.5, 8.8)
                    
            new_h = self.compute_tangents(indices, new_p, new_r, centers, start_node_idx)
            trial_pts = get_points(new_h, new_p, new_r)
            t_len, c_cost, c_col = self.calc_cost_and_collisions(indices, trial_pts, centers, start_node_idx, penalty_base=500000.0, penalty_slope=500000.0)
            
            if best_col == 0 and c_col > 0:
                continue
            trial_cost = t_len + c_cost
            delta = trial_cost - curr_cost
            if delta < 0 or (t_curr > 0 and random.random() < np.exp(-delta / t_curr)):
                headings, phi_angles, radii = new_h, new_p, new_r
                curr_cost = trial_cost
                if (c_col < best_col) or (c_col == best_col and trial_cost < best_cost):
                    best_h, best_p, best_r = np.array(headings), np.array(phi_angles), np.array(radii)
                    best_cost = trial_cost
                    best_col = c_col
                    
        final_pts = get_points(best_h, best_p, best_r)
        
        # 5. Advanced Collision Eraser (Dense Radial + Extended Heading + Joint Flank Search)
        for pass_round in range(8):
            has_col = False
            for i in range(n):
                u = best_indices[i]
                v = best_indices[(i + 1) % n]
                hit, _, _ = self.check_segment_collision(final_pts[u], final_pts[v], centers, start_node_idx, safe_r=5.12)
                if hit:
                    has_col = True
                    repaired = False
                    
                    # Pass 1: Single Node Optimization on v then u
                    for fix_target in [v, u]:
                        if fix_target == start_node_idx: continue
                        pos = best_indices.index(fix_target)
                        prev_n = best_indices[(pos - 1) % n]
                        next_n = best_indices[(pos + 1) % n]
                        c = centers[fix_target]
                        
                        best_candidate = None
                        best_cand_len = 1e9
                        
                        # Test angles (36 uniform) and expanded radii (up to 8.8 km)
                        for phi in np.linspace(0, 2*np.pi, 36, endpoint=False):
                            for r in [6.8, 7.5, 8.2, 8.8]:
                                cand_xy = c + r * np.array([np.cos(phi), np.sin(phi)])
                                # Distance to other obstacles
                                if any(np.linalg.norm(cand_xy - centers[o]) < 5.25 for o in range(n) if o != fix_target and o != start_node_idx):
                                    continue
                                    
                                v_next = final_pts[next_n][:2] - cand_xy
                                v_prev = cand_xy - final_pts[prev_n][:2]
                                d_next = v_next / (np.linalg.norm(v_next) + 1e-6)
                                d_prev = v_prev / (np.linalg.norm(v_prev) + 1e-6)
                                d_avg = d_next + d_prev
                                d_avg /= (np.linalg.norm(d_avg) + 1e-6)
                                
                                base_headings = [
                                    np.arctan2(d_next[1], d_next[0]),
                                    np.arctan2(d_prev[1], d_prev[0]),
                                    np.arctan2(d_avg[1], d_avg[0]),
                                    phi + np.pi/2, phi - np.pi/2, phi
                                ]
                                test_headings = []
                                for bh in base_headings:
                                    test_headings.extend([bh, bh - 0.20, bh + 0.20])
                                    
                                for h in test_headings:
                                    cand_p = np.array([cand_xy[0], cand_xy[1], h])
                                    hit_in, l_in, _ = self.check_segment_collision(final_pts[prev_n], cand_p, centers, start_node_idx, safe_r=5.08)
                                    if hit_in: continue
                                    hit_out, l_out, _ = self.check_segment_collision(cand_p, final_pts[next_n], centers, start_node_idx, safe_r=5.08)
                                    if hit_out: continue
                                    
                                    if l_in + l_out < best_cand_len:
                                        best_cand_len = l_in + l_out
                                        best_candidate = cand_p
                                        
                        if best_candidate is not None:
                            final_pts[fix_target] = best_candidate
                            repaired = True
                            break
                            
                    # Pass 2: Joint Pair Optimization if single fix fails
                    if not repaired and u != start_node_idx and v != start_node_idx:
                        pos_u = best_indices.index(u)
                        pos_v = best_indices.index(v)
                        prev_u = best_indices[(pos_u - 1) % n]
                        next_v = best_indices[(pos_v + 1) % n]
                        cu, cv = centers[u], centers[v]
                        
                        best_pair = None
                        best_pair_len = 1e9
                        
                        # Test both standard and wide radii
                        for r_pair in [7.5, 8.5, 8.8]:
                            for phi_u in np.linspace(0, 2*np.pi, 20, endpoint=False):
                                for phi_v in np.linspace(0, 2*np.pi, 20, endpoint=False):
                                    xy_u = cu + r_pair * np.array([np.cos(phi_u), np.sin(phi_u)])
                                    xy_v = cv + r_pair * np.array([np.cos(phi_v), np.sin(phi_v)])
                                    
                                    if any(np.linalg.norm(xy_u - centers[o]) < 5.25 for o in range(n) if o != u and o != start_node_idx):
                                        continue
                                    if any(np.linalg.norm(xy_v - centers[o]) < 5.25 for o in range(n) if o != v and o != start_node_idx):
                                        continue
                                        
                                    v_in_u = xy_u - final_pts[prev_u][:2]
                                    v_uv = xy_v - xy_u
                                    v_out_v = final_pts[next_v][:2] - xy_v
                                    
                                    t_u = v_in_u / np.linalg.norm(v_in_u) + v_uv / np.linalg.norm(v_uv)
                                    t_v = v_uv / np.linalg.norm(v_uv) + v_out_v / np.linalg.norm(v_out_v)
                                    th_chord = np.arctan2(v_uv[1], v_uv[0])
                                    
                                    for hu in [np.arctan2(t_u[1], t_u[0]), th_chord]:
                                        for hv in [np.arctan2(t_v[1], t_v[0]), th_chord]:
                                            cand_u = np.array([xy_u[0], xy_u[1], hu])
                                            cand_v = np.array([xy_v[0], xy_v[1], hv])
                                            
                                            h1, l1, _ = self.check_segment_collision(final_pts[prev_u], cand_u, centers, start_node_idx, safe_r=5.05)
                                            if h1: continue
                                            h2, l2, _ = self.check_segment_collision(cand_u, cand_v, centers, start_node_idx, safe_r=5.05)
                                            if h2: continue
                                            h3, l3, _ = self.check_segment_collision(cand_v, final_pts[next_v], centers, start_node_idx, safe_r=5.05)
                                            if h3: continue
                                            
                                            if l1 + l2 + l3 < best_pair_len:
                                                best_pair_len = l1 + l2 + l3
                                                best_pair = (cand_u, cand_v)
                                                break
                                        if best_pair is not None: break
                                    if best_pair is not None: break
                                if best_pair is not None: break
                            if best_pair is not None: break
                            
                        if best_pair is not None:
                            final_pts[u], final_pts[v] = best_pair
            if not has_col: break
            
        final_len = 0.0
        final_col = 0
        for i in range(n):
            u = best_indices[i]
            v = best_indices[(i + 1) % n]
            hit, l, min_d = self.check_segment_collision(final_pts[u], final_pts[v], centers, start_node_idx, safe_r=5.0)
            final_len += l
            if hit: final_col += 1
                
        return best_indices, final_pts, final_len, final_col

# =============================================================================
# 繪圖工具

# =============================================================================

def draw_single_panel(ax, centers, indices, points, start_node_idx, title, sub_title, cost_calculator):
    ax.set_facecolor('white')
    ax.set_title(f"{title}\n{sub_title}", fontsize=11, fontweight='bold', pad=12, color='#0f172a')
    ax.set_xlabel("East (km)", fontsize=9, fontweight='bold')
    ax.set_ylabel("North (km)", fontsize=9, fontweight='bold')
    ax.grid(True, linestyle='--', color='#cbd5e1', alpha=0.5)
    
    # 畫障礙圈與甜甜圈
    for i, c in enumerate(centers):
        if i == start_node_idx: continue
        ax.add_patch(Circle((c[0], c[1]), 5.0, facecolor='#fee2e2', edgecolor='#ef4444', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1))
        ax.add_patch(Circle((c[0], c[1]), 9.0, facecolor='#e0f2fe', edgecolor='#38bdf8', linestyle=':', linewidth=0.8, alpha=0.35, zorder=1))
        ax.scatter(c[0], c[1], c='#dc2626', s=15, zorder=2)
        
    # 畫起點與觀測點
    ax.scatter(centers[start_node_idx, 0], centers[start_node_idx, 1], c='#eab308', s=120, marker='*', edgecolors='#854d0e', zorder=5, label='Depot')
    ax.scatter(points[:, 0], points[:, 1], c='#16a34a', s=25, edgecolors='#14532d', zorder=4, label='Waypoints')
    
    # 畫航向箭頭
    u = np.cos(points[:, 2]) * 2.2
    v = np.sin(points[:, 2]) * 2.2
    ax.quiver(points[:, 0], points[:, 1], u, v, color='#0f172a', scale=1, scale_units='xy', angles='xy', width=0.005, headwidth=4, headlength=4, zorder=6)
    
    # 畫真實 Dubins 曲線
    n = len(indices)
    for i in range(n):
        idx1 = indices[i]
        idx2 = indices[(i + 1) % n]
        p1 = points[idx1]
        p2 = points[idx2]
        
        t, p, q, mode, _ = cost_calculator._plan_dubins(p1, p2)
        if mode is None:
            px = np.linspace(p1[0], p2[0], 25)
            py = np.linspace(p1[1], p2[1], 25)
        else:
            px, py = cost_calculator._interpolate(p1, t, p, q, mode, step_size=0.15)
            
        ax.plot(px, py, color='#1d4ed8', linewidth=1.8, alpha=0.85, zorder=3)
        
    ax.set_aspect('equal', 'box')


# =============================================================================
# 主執行比對腳本
# =============================================================================

def run_benchmark():
    print("=================================================================")
    print(" [BENCHMARK] Starting DTSPN 3-Version Benchmark (Fixed Seed)")
    print("=================================================================")
    
    # 1. 固定環境隨機種子
    FIXED_SEED = 42
    np.random.seed(FIXED_SEED)
    random.seed(FIXED_SEED)
    
    # 生成 25 艘靜態敵艦 (50~150km, 0~100km)
    n_static = 25
    static_centers = np.column_stack([
        np.random.uniform(50, 150, n_static),
        np.random.uniform(0, 100, n_static)
    ])
    
    # 起點設定 (最西北角)
    norm_x = (static_centers[:, 0] - 50) / 100.0
    norm_y = (static_centers[:, 1] - 0) / 100.0
    start_node_idx = int(np.argmin(norm_x - norm_y))
    print(f"固定起點基地索引: {start_node_idx}，位置: {static_centers[start_node_idx]}")
    
    # 預先生成 5 個動態新目標 (保證三個版本遇到的動態目標順序與位置 100% 相同)
    dynamic_targets = [
        np.random.uniform([50, 0], [150, 100]) for _ in range(5)
    ]
    
    # 實例化 3 個求解器
    v1_solver = Version1_Baseline()
    v2_solver = Version2_SmartInit()
    v3_solver = Version3_TangentSmoothing()
    v5_solver = Version5_ZeroCollisionFast()
    
    results = {}
    
    # -------------------------------------------------------------------------
    # Stage 0: 25 個靜態目標規劃
    # -------------------------------------------------------------------------
    print("\n>>> 正在執行 Stage 0 (25 靜態目標) 優化...")
    
    # V1
    t0 = time.time()
    v1_idx_0, v1_pts_0, v1_len_0, v1_col_0 = v1_solver.solve(static_centers, start_node_idx)
    v1_time_0 = time.time() - t0
    print(f" [V1 學長原版 Baseline]        航程: {v1_len_0:.2f} km | 碰撞: {v1_col_0} 次 | 耗時: {v1_time_0:.2f}s")
    
    # V2
    t0 = time.time()
    v2_idx_0, v2_pts_0, v2_len_0, v2_col_0 = v2_solver.solve(static_centers, start_node_idx)
    v2_time_0 = time.time() - t0
    print(f" [V2 2-opt初解+高懲罰防碰撞]  航程: {v2_len_0:.2f} km | 碰撞: {v2_col_0} 次 | 耗時: {v2_time_0:.2f}s")
    
    # V3
    t0 = time.time()
    v3_idx_0, v3_pts_0, v3_len_0, v3_col_0 = v3_solver.solve(static_centers, start_node_idx)
    v3_time_0 = time.time() - t0
    print(f" [V3 切線平滑+外切流線]       航程: {v3_len_0:.2f} km | 碰撞: {v3_col_0} 次 | 耗時: {v3_time_0:.2f}s")

    # V5
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    t0 = time.time()
    v5_idx_0, v5_pts_0, v5_len_0, v5_col_0 = v5_solver.solve(static_centers, start_node_idx)
    v5_time_0 = time.time() - t0
    print(f" [V5 Transformer 嚴格零碰撞]       航程: {v5_len_0:.2f} km | 碰撞: {v5_col_0} 次 | 耗時: {v5_time_0:.2f}s")

    
    # -------------------------------------------------------------------------
    # Stage 5: 逐一加入 5 艘動態敵艦後的最終規劃 (共 30 目標)
    # -------------------------------------------------------------------------
    print("\n>>> 正在執行 Stage 1~5 動態重規劃 (最終 30 目標)...")
    
    # V1 動態重規劃
    v1_centers = np.copy(static_centers)
    for dyn_tgt in dynamic_targets:
        v1_centers = np.vstack([v1_centers, dyn_tgt])
    v1_idx_5, v1_pts_5, v1_len_5, v1_col_5 = v1_solver.solve(v1_centers, start_node_idx)
    print(f" [V1 學長原版 Baseline - Stage 5]        航程: {v1_len_5:.2f} km | 碰撞: {v1_col_5} 次")
    
    # V2 動態重規劃
    v2_centers = np.copy(static_centers)
    for dyn_tgt in dynamic_targets:
        v2_centers = np.vstack([v2_centers, dyn_tgt])
    v2_idx_5, v2_pts_5, v2_len_5, v2_col_5 = v2_solver.solve(v2_centers, start_node_idx)
    print(f" [V2 2-opt初解+高懲罰防碰撞 - Stage 5]  航程: {v2_len_5:.2f} km | 碰撞: {v2_col_5} 次")
    
    # V3 動態重規劃
    v3_centers = np.copy(static_centers)
    for dyn_tgt in dynamic_targets:
        v3_centers = np.vstack([v3_centers, dyn_tgt])
    v3_idx_5, v3_pts_5, v3_len_5, v3_col_5 = v3_solver.solve(v3_centers, start_node_idx)
    print(f" [V3 切線平滑+外切流線 - Stage 5]       航程: {v3_len_5:.2f} km | 碰撞: {v3_col_5} 次")

    # V5 動態重規劃
    v5_centers = np.copy(static_centers)
    for dyn_tgt in dynamic_targets:
        v5_centers = np.vstack([v5_centers, dyn_tgt])
    
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    t0 = time.time()
    v5_idx_5, v5_pts_5, v5_len_5, v5_col_5 = v5_solver.solve(v5_centers, start_node_idx)
    v5_time_5 = time.time() - t0
    print(f" [V5 Transformer 嚴格零碰撞 - Stage 5]  航程: {v5_len_5:.2f} km | 碰撞: {v5_col_5} 次 | 耗時: {v5_time_5:.2f}s")


    # =========================================================================
    # 輸出 Stage 0 與 Stage 5 的 3 欄橫向並排對比圖
    # =========================================================================
    print("\n>>> 正在繪製並排對比視覺化大圖...")
    cost_calc = DubinsCost(2.0)
    
    # --- 圖 1: Stage 0 對比圖 ---
    fig_0, axes_0 = plt.subplots(2, 2, figsize=(16, 14), dpi=200)
    axes_0 = axes_0.flatten()
    fig_0.suptitle("DTSP Algorithm Benchmark: Stage 0 (25 Static Targets)", fontsize=16, fontweight='bold', y=0.98)
    
    draw_single_panel(axes_0[0], static_centers, v1_idx_0, v1_pts_0, start_node_idx, 
                      "Version 1: Baseline (Senior)", f"Length: {v1_len_0:.2f} km  |  Collisions: {v1_col_0}", cost_calc)
    draw_single_panel(axes_0[1], static_centers, v2_idx_0, v2_pts_0, start_node_idx, 
                      "Version 2: 2-Opt & Penalty", f"Length: {v2_len_0:.2f} km  |  Collisions: {v2_col_0}", cost_calc)
    draw_single_panel(axes_0[2], static_centers, v3_idx_0, v3_pts_0, start_node_idx, 
                      "Version 3: Tangent Smoothing (Ours)", f"Length: {v3_len_0:.2f} km  |  Collisions: {v3_col_0}", cost_calc)
    draw_single_panel(axes_0[3], static_centers, v5_idx_0, v5_pts_0, start_node_idx, 
                      "Version 5: NN Strict (0-Collision)", f"Length: {v5_len_0:.2f} km  |  Collisions: {v5_col_0}", cost_calc)

    
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])
    plt.subplots_adjust(top=0.85, wspace=0.18)
    fig_0.savefig("comparison_stage_0.png", bbox_inches='tight', facecolor='white')
    plt.close(fig_0)
    print("[SUCCESS] Saved Stage 0 comparison: comparison_stage_0.png")
    
    # --- 圖 2: Stage 5 對比圖 ---
    fig_5, axes_5 = plt.subplots(2, 2, figsize=(16, 14), dpi=200)
    axes_5 = axes_5.flatten()
    fig_5.suptitle("DTSP Algorithm Benchmark: Stage 5 (30 Total Targets)", fontsize=16, fontweight='bold', y=0.98)
    
    draw_single_panel(axes_5[0], v1_centers, v1_idx_5, v1_pts_5, start_node_idx, 
                      "Version 1: Baseline (Senior)", f"Length: {v1_len_5:.2f} km  |  Collisions: {v1_col_5}", cost_calc)
    draw_single_panel(axes_5[1], v2_centers, v2_idx_5, v2_pts_5, start_node_idx, 
                      "Version 2: 2-Opt & Penalty", f"Length: {v2_len_5:.2f} km  |  Collisions: {v2_col_5}", cost_calc)
    draw_single_panel(axes_5[2], v3_centers, v3_idx_5, v3_pts_5, start_node_idx, 
                      "Version 3: Tangent Smoothing (Ours)", f"Length: {v3_len_5:.2f} km  |  Collisions: {v3_col_5}", cost_calc)

    draw_single_panel(axes_5[3], v5_centers, v5_idx_5, v5_pts_5, start_node_idx, 
                      "Version 5: NN Strict (0-Collision)", f"Length: {v5_len_5:.2f} km  |  Collisions: {v5_col_5}", cost_calc)
    
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])
    plt.subplots_adjust(top=0.85, wspace=0.18)
    fig_5.savefig("comparison_stage_5.png", bbox_inches='tight', facecolor='white')
    plt.close(fig_5)
    print("[SUCCESS] Saved Stage 5 comparison: comparison_stage_5.png")

    # --- 圖 3: 數據量化綜合評估圖 (長條圖) ---
    fig_bar, ax_bar = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
    versions = ['V1 Baseline\n(Senior)', 'V2 2-Opt &\nPenalty', 'V3 Tangent\nSmoothing', 'V5 NN Strict\n(0-Collision)']
    
    # 航程對比
    stage0_lens = [v1_len_0, v2_len_0, v3_len_0, v5_len_0]
    stage5_lens = [v1_len_5, v2_len_5, v3_len_5, v5_len_5]
    x = np.arange(len(versions))
    width = 0.35
    
    b1 = ax_bar[0].bar(x - width/2, stage0_lens, width, label='Stage 0 (25 targets)', color='#93c5fd', edgecolor='#1d4ed8')
    b2 = ax_bar[0].bar(x + width/2, stage5_lens, width, label='Stage 5 (30 targets)', color='#60a5fa', edgecolor='#1e40af')
    ax_bar[0].set_ylabel('Trajectory Length (km)', fontweight='bold')
    ax_bar[0].set_title('Dubins Trajectory Length Comparison', fontweight='bold')
    ax_bar[0].set_xticks(x)
    ax_bar[0].set_xticklabels(versions, fontweight='bold')
    ax_bar[0].legend()
    ax_bar[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in b1 + b2:
        yval = bar.get_height()
        ax_bar[0].text(bar.get_x() + bar.get_width()/2.0, yval + 10, f'{yval:.1f} km', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 碰撞次數對比
    stage0_cols = [v1_col_0, v2_col_0, v3_col_0, v5_col_0]
    stage5_cols = [v1_col_5, v2_col_5, v3_col_5, v5_col_5]
    c1 = ax_bar[1].bar(x - width/2, stage0_cols, width, label='Stage 0 Collisions', color='#fca5a5', edgecolor='#dc2626')
    c2 = ax_bar[1].bar(x + width/2, stage5_cols, width, label='Stage 5 Collisions', color='#ef4444', edgecolor='#991b1b')
    ax_bar[1].set_ylabel('Collision Count (Red Zone Penetration)', fontweight='bold')
    ax_bar[1].set_title('Obstacle Avoidance & Safety Verification', fontweight='bold')
    ax_bar[1].set_xticks(x)
    ax_bar[1].set_xticklabels(versions, fontweight='bold')
    ax_bar[1].legend()
    ax_bar[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in c1 + c2:
        yval = bar.get_height()
        ax_bar[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{int(yval)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig_bar.savefig("comparison_metrics.png", bbox_inches='tight', facecolor='white')
    plt.close(fig_bar)
    print("[SUCCESS] Saved Metrics Chart: comparison_metrics.png")
    
    print("\n=================================================================")
    print(" [BENCHMARK COMPLETED SUCCESSFULLY]")
    print("=================================================================")


if __name__ == '__main__':
    run_benchmark()
