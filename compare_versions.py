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
    
    # 畫 Dubins 航跡
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
            px, py = cost_calculator._interpolate(p1, t, p, q, mode, step_size=0.5)
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

    # =========================================================================
    # 輸出 Stage 0 與 Stage 5 的 3 欄橫向並排對比圖
    # =========================================================================
    print("\n>>> 正在繪製並排對比視覺化大圖...")
    cost_calc = DubinsCost(2.0)
    
    # --- 圖 1: Stage 0 對比圖 ---
    fig_0, axes_0 = plt.subplots(1, 3, figsize=(22, 7.5), dpi=200)
    fig_0.suptitle("DTSP Algorithm Benchmark: Stage 0 (25 Static Targets)", fontsize=16, fontweight='bold', y=0.98)
    
    draw_single_panel(axes_0[0], static_centers, v1_idx_0, v1_pts_0, start_node_idx, 
                      "Version 1: Baseline (Senior)", f"Length: {v1_len_0:.2f} km  |  Collisions: {v1_col_0}", cost_calc)
    draw_single_panel(axes_0[1], static_centers, v2_idx_0, v2_pts_0, start_node_idx, 
                      "Version 2: 2-Opt & Penalty", f"Length: {v2_len_0:.2f} km  |  Collisions: {v2_col_0}", cost_calc)
    draw_single_panel(axes_0[2], static_centers, v3_idx_0, v3_pts_0, start_node_idx, 
                      "Version 3: Tangent Smoothing (Ours)", f"Length: {v3_len_0:.2f} km  |  Collisions: {v3_col_0}", cost_calc)
    
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])
    plt.subplots_adjust(top=0.85, wspace=0.18)
    fig_0.savefig("comparison_stage_0.png", bbox_inches='tight', facecolor='white')
    plt.close(fig_0)
    print("[SUCCESS] Saved Stage 0 comparison: comparison_stage_0.png")
    
    # --- 圖 2: Stage 5 對比圖 ---
    fig_5, axes_5 = plt.subplots(1, 3, figsize=(22, 7.5), dpi=200)
    fig_5.suptitle("DTSP Algorithm Benchmark: Stage 5 (30 Total Targets)", fontsize=16, fontweight='bold', y=0.98)
    
    draw_single_panel(axes_5[0], v1_centers, v1_idx_5, v1_pts_5, start_node_idx, 
                      "Version 1: Baseline (Senior)", f"Length: {v1_len_5:.2f} km  |  Collisions: {v1_col_5}", cost_calc)
    draw_single_panel(axes_5[1], v2_centers, v2_idx_5, v2_pts_5, start_node_idx, 
                      "Version 2: 2-Opt & Penalty", f"Length: {v2_len_5:.2f} km  |  Collisions: {v2_col_5}", cost_calc)
    draw_single_panel(axes_5[2], v3_centers, v3_idx_5, v3_pts_5, start_node_idx, 
                      "Version 3: Tangent Smoothing (Ours)", f"Length: {v3_len_5:.2f} km  |  Collisions: {v3_col_5}", cost_calc)
    
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])
    plt.subplots_adjust(top=0.85, wspace=0.18)
    fig_5.savefig("comparison_stage_5.png", bbox_inches='tight', facecolor='white')
    plt.close(fig_5)
    print("[SUCCESS] Saved Stage 5 comparison: comparison_stage_5.png")

    # --- 圖 3: 數據量化綜合評估圖 (長條圖) ---
    fig_bar, ax_bar = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
    versions = ['V1 Baseline\n(Senior)', 'V2 2-Opt &\nPenalty', 'V3 Tangent\nSmoothing (Ours)']
    
    # 航程對比
    stage0_lens = [v1_len_0, v2_len_0, v3_len_0]
    stage5_lens = [v1_len_5, v2_len_5, v3_len_5]
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
    stage0_cols = [v1_col_0, v2_col_0, v3_col_0]
    stage5_cols = [v1_col_5, v2_col_5, v3_col_5]
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
