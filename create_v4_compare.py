import re

with open("compare_versions.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Add torch import
code = code.replace("import numpy as np", "import numpy as np\nimport torch\nimport os")

# 2. Add V4 Class after V3 class
v4_class = """
# -----------------------------------------------------------------------------
# 版本 4: Transformer V4 + SA Physical Controller
# -----------------------------------------------------------------------------
class Version4_TransformerAndSA(Version3_TangentSmoothing):
    def __init__(self, model_path="transformer_checkpoints/best_model.pt", turning_radius=2.0, obs_min=5.0, obs_max=9.0):
        super().__init__(turning_radius, obs_min, obs_max)
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
                print(f"[WARN] V4 Model not found at {model_path}!")
        except Exception as e:
            print(f"[ERROR] Failed to load V4 Model: {e}")
            self.policy = None

    def solve(self, centers, start_node_idx, max_iter=500, cooling_rate=0.996):
        n = len(centers)
        
        # --- 1. 使用 Transformer 推論初始 Sequence ---
        if self.policy is not None:
            coords = np.zeros((n, 2), dtype=np.float32)
            coords[:, 0] = (centers[:, 0] - 50.0) / 100.0
            coords[:, 1] = (centers[:, 1] - 0.0) / 100.0
            obs_radii = np.ones(n, dtype=np.float32) * (self.obs_max_radius / 100.0)
            obs_radii[start_node_idx] = 0.0
            
            coords_t = torch.tensor(coords, device=self.device).unsqueeze(0)
            obs_radii_t = torch.tensor(obs_radii, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                tours, _ = self.policy(coords_t, obs_radii_t, greedy=True, start_city=start_node_idx)
                indices = tours[0].cpu().numpy().tolist()
        else:
            indices = build_2opt_euclidean_tour(centers, start_node_idx)
            
        # --- 2. 使用 SA 優化 Heading, Phi, Radius (鎖定 indices 不做大幅更動) ---
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
            if np.linalg.norm(tangent) < 1e-3: out_n = vec_to_c
            else:
                out_n = np.array([-tangent[1], tangent[0]])
                if np.dot(out_n, vec_to_c) < 0: out_n = -out_n
            phi_angles[target_idx] = np.arctan2(out_n[1], out_n[0]) if np.linalg.norm(out_n) > 1e-3 else np.arctan2(vec_to_c[1], vec_to_c[0])
            
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
        t_len, c_cost, c_col = self.calc_cost_and_collisions(indices, curr_pts, centers, start_node_idx, penalty_base=50000.0, penalty_slope=50000.0)
        curr_cost = t_len + c_cost
        
        best_indices = list(indices)
        best_h, best_p, best_r = np.array(headings), np.array(phi_angles), np.array(radii)
        best_cost = curr_cost
        best_len, best_col = t_len, c_col
        
        temp = 200.0
        # SA 微調 (只做 500 次，發揮 Transformer 速度優勢)
        for step in range(max_iter):
            t_curr = temp * (cooling_rate ** step)
            new_idx = list(indices)
            new_p, new_r = np.array(phi_angles), np.array(radii)
            
            r_val = random.random()
            # 降低 sequence 變更機率，因為 transformer 已經很準了
            if r_val < 0.02 and n >= 4:
                idx1, idx2 = random.sample(range(1, n), 2)
                if idx1 > idx2: idx1, idx2 = idx2, idx1
                new_idx = indices[:idx1] + indices[idx1:idx2+1][::-1] + indices[idx2+1:]
            elif r_val < 0.60:
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
"""
code = code.replace("# =============================================================================\n# 繪圖工具", v4_class)

# 3. Modify run_benchmark instances
code = code.replace("v3_solver = Version3_TangentSmoothing()", "v3_solver = Version3_TangentSmoothing()\n    v4_solver = Version4_TransformerAndSA()")

# Stage 0 block
v4_stage0 = """
    # V4
    t0 = time.time()
    v4_idx_0, v4_pts_0, v4_len_0, v4_col_0 = v4_solver.solve(static_centers, start_node_idx)
    v4_time_0 = time.time() - t0
    print(f" [V4 Transformer+SA 物理微調]       航程: {v4_len_0:.2f} km | 碰撞: {v4_col_0} 次 | 耗時: {v4_time_0:.2f}s")
"""
code = code.replace("print(f\" [V3 切線平滑+外切流線]       航程: {v3_len_0:.2f} km | 碰撞: {v3_col_0} 次 | 耗時: {v3_time_0:.2f}s\")", 
                    "print(f\" [V3 切線平滑+外切流線]       航程: {v3_len_0:.2f} km | 碰撞: {v3_col_0} 次 | 耗時: {v3_time_0:.2f}s\")\n" + v4_stage0)

# Stage 5 block
v4_stage5 = """
    # V4 動態重規劃
    v4_centers = np.copy(static_centers)
    for dyn_tgt in dynamic_targets:
        v4_centers = np.vstack([v4_centers, dyn_tgt])
    
    t0 = time.time()
    v4_idx_5, v4_pts_5, v4_len_5, v4_col_5 = v4_solver.solve(v4_centers, start_node_idx)
    v4_time_5 = time.time() - t0
    print(f" [V4 Transformer+SA 物理微調 - Stage 5]  航程: {v4_len_5:.2f} km | 碰撞: {v4_col_5} 次 | 耗時: {v4_time_5:.2f}s")
"""
code = code.replace("print(f\" [V3 切線平滑+外切流線 - Stage 5]       航程: {v3_len_5:.2f} km | 碰撞: {v3_col_5} 次\")",
                    "print(f\" [V3 切線平滑+外切流線 - Stage 5]       航程: {v3_len_5:.2f} km | 碰撞: {v3_col_5} 次\")\n" + v4_stage5)

# 4. Update plot layouts from 1x3 to 2x2
code = code.replace("fig_0, axes_0 = plt.subplots(1, 3, figsize=(22, 7.5), dpi=200)", "fig_0, axes_0 = plt.subplots(2, 2, figsize=(16, 14), dpi=200)\n    axes_0 = axes_0.flatten()")
code = code.replace("fig_5, axes_5 = plt.subplots(1, 3, figsize=(22, 7.5), dpi=200)", "fig_5, axes_5 = plt.subplots(2, 2, figsize=(16, 14), dpi=200)\n    axes_5 = axes_5.flatten()")

# Plot 1
code = code.replace("cost_calc)\n    \n    plt.tight_layout", "cost_calc)\n    draw_single_panel(axes_0[3], static_centers, v4_idx_0, v4_pts_0, start_node_idx, \n                      \"Version 4: Transformer + SA (Fast & Safe)\", f\"Length: {v4_len_0:.2f} km  |  Collisions: {v4_col_0}\", cost_calc)\n    \n    plt.tight_layout")
# Plot 2
code = code.replace("cost_calc)\n    \n    plt.tight_layout", "cost_calc)\n    draw_single_panel(axes_5[3], v4_centers, v4_idx_5, v4_pts_5, start_node_idx, \n                      \"Version 4: Transformer + SA (Fast & Safe)\", f\"Length: {v4_len_5:.2f} km  |  Collisions: {v4_col_5}\", cost_calc)\n    \n    plt.tight_layout")

# 5. Update bar charts
code = code.replace("versions = ['V1 Baseline\\n(Senior)', 'V2 2-Opt &\\nPenalty', 'V3 Tangent\\nSmoothing (Ours)']", "versions = ['V1 Baseline\\n(Senior)', 'V2 2-Opt &\\nPenalty', 'V3 Tangent\\nSmoothing', 'V4 Transformer\\n+ SA']")
code = code.replace("stage0_lens = [v1_len_0, v2_len_0, v3_len_0]", "stage0_lens = [v1_len_0, v2_len_0, v3_len_0, v4_len_0]")
code = code.replace("stage5_lens = [v1_len_5, v2_len_5, v3_len_5]", "stage5_lens = [v1_len_5, v2_len_5, v3_len_5, v4_len_5]")
code = code.replace("stage0_cols = [v1_col_0, v2_col_0, v3_col_0]", "stage0_cols = [v1_col_0, v2_col_0, v3_col_0, v4_col_0]")
code = code.replace("stage5_cols = [v1_col_5, v2_col_5, v3_col_5]", "stage5_cols = [v1_col_5, v2_col_5, v3_col_5, v4_col_5]")

with open("compare_versions_v4.py", "w", encoding="utf-8") as f:
    f.write(code)
print("v4 patching script created.")
