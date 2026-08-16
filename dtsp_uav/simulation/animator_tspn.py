import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from core.dubins import DubinsCost

class AnimatorTSPN:
    def __init__(self, env, turning_radius=2.0, obstacle_radius=5.0, obs_max_radius=9.0):
        """
        DTSPN 視覺化器，採用適合論文發表的白色高對比學術風格。
        """
        self.env = env
        self.turning_radius = turning_radius
        self.obstacle_radius = obstacle_radius
        self.obs_max_radius = obs_max_radius
        self.cost_calculator = DubinsCost(turning_radius)
        self.n_static = len(env.targets)

    def _setup_axes(self, ax):
        """設定適合論文發表的明亮色系座標軸"""
        ax.set_facecolor('white')  # 純白背景
        ax.set_xlim(self.env.x_range[0] - 15, self.env.x_range[1] + 15)
        ax.set_ylim(self.env.y_range[0] - 15, self.env.y_range[1] + 15)
        ax.set_aspect('equal')
        
        # 設定軸標籤與格線 (深色字體，淡灰色格線)
        ax.set_xlabel("East (km)", color='#1e293b', fontsize=11, fontweight='bold')
        ax.set_ylabel("North (km)", color='#1e293b', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#334155', labelsize=10)
        ax.grid(True, linestyle='--', color='#cbd5e1', alpha=0.5, zorder=0)
        
        # 邊框設定為深灰色
        for spine in ax.spines.values():
            spine.set_color('#64748b')
            spine.set_linewidth(1.0)

    def _interpolate_dubins_path(self, p1, p2):
        """計算兩點間的插值航跡"""
        self.cost_calculator.R = self.turning_radius
        path_x, path_y = self.cost_calculator.get_path_points(p1, p2)
        return path_x, path_y

    def _draw_state(self, ax, centers, points, route, start_node_idx, title, phi_angles, radii):
        """繪製單個狀態"""
        ax.clear()
        self._setup_axes(ax)
        ax.set_title(title, color='#0f172a', fontsize=12, fontweight='bold', pad=12)

        # 1. 繪製所有目標艦艇的限制區 (避障區與觀測帶)
        for i in range(len(centers)):
            if i == start_node_idx:
                continue
            
            c = centers[i]
            # 避障區 (5km 半徑，淡紅填充 + 紅色虛線)
            obs_circle = patches.Circle(c, self.obstacle_radius, 
                                        facecolor='#fee2e2', edgecolor='#ef4444', 
                                        alpha=0.6, linestyle='--', linewidth=1.0, zorder=1)
            ax.add_patch(obs_circle)
            
            # 觀測帶甜甜圈 (5km ~ 9km，淡藍填充 + 藍色細點線)
            annulus = patches.Wedge(c, self.obs_max_radius, 0, 360, 
                                    width=self.obs_max_radius - self.obstacle_radius,
                                    facecolor='#e0f2fe', edgecolor='#38bdf8', 
                                    alpha=0.5, linestyle=':', linewidth=0.8, zorder=0)
            ax.add_patch(annulus)
            
            # 敵艦中心點 (小紅點)
            ax.scatter(c[0], c[1], c='#dc2626', s=25, label='Enemy Target Center' if i == 1 else "", zorder=3)

        # 2. 標記出各階段的點集 (區分靜態與動態)
        # 靜態目標的拜訪點 (深綠色)
        ax.scatter(points[:self.n_static, 0], points[:self.n_static, 1], 
                   c='#16a34a', s=40, edgecolors='#14532d', label='Visited Points (Static)', zorder=5)
        
        # 動態目標中心點及其拜訪點
        if len(centers) > self.n_static:
            # 標記動態敵艦中心點為紫色
            ax.scatter(centers[self.n_static:, 0], centers[self.n_static:, 1],
                       c='#7c3aed', s=30, edgecolors='#4c1d95', label='Dynamic Target Centers', zorder=4)
            # 標記動態目標的拜訪點 (淺紫)
            ax.scatter(points[self.n_static:, 0], points[self.n_static:, 1],
                       c='#ddd6fe', s=45, edgecolors='#4c1d95', label='Visited Points (Dynamic)', zorder=5)
            # 特別標出最新加入的那一艘 (黃色外圈大點做強調)
            ax.scatter(centers[-1, 0], centers[-1, 1],
                       c='#f59e0b', s=100, marker='o', edgecolors='#b45309', label='Newest Target Center', zorder=4)

        # 3. 標記起點 (Depot / Launch Base - 顯眼的金色星星)
        ax.scatter(centers[start_node_idx, 0], centers[start_node_idx, 1], 
                   c='#eab308', s=180, marker='*', edgecolors='#854d0e', linewidths=1.2, label='UAV Launch Base', zorder=6)

        # 4. 繪製 Dubins 閉環航跡 (使用經典的深藍色加粗實線，確保論文列印清晰)
        closed_idx = route.get_closed_path()
        if len(closed_idx) > 1:
            for i in range(len(closed_idx) - 1):
                u, v = closed_idx[i], closed_idx[i+1]
                px, py = self._interpolate_dubins_path(points[u], points[v])
                ax.plot(px, py, color='#1e40af', linewidth=2.0, alpha=0.9, label='UAV Trajectory' if i == 0 else "", zorder=7)

        # 5. 繪製拜訪時的 UAV 航向箭頭 (改為深色箭頭以利辨識)
        u = np.cos(points[:, 2])
        v = np.sin(points[:, 2])
        ax.quiver(points[:, 0], points[:, 1], u, v, color='#0f172a', 
                  pivot='middle', width=0.004, headwidth=4, alpha=0.8, zorder=8)

        # 6. 圖例 (去重，改為白底黑邊，適合學術風格)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', 
                  facecolor='white', edgecolor='#64748b', labelcolor='#0f172a', fontsize=9, framealpha=0.9)

    def render_snapshots(self, snapshots, save_dir="."):
        """
        渲染並保存多個階段的規劃結果為 PNG。
        :param snapshots: 包含 (centers, points, route, start_idx, phi, radii) 的列表
        """
        print("開始生成論文標準視覺化快照...")
        for i, (centers, points, route, start, phi, radii) in enumerate(snapshots):
            fig, ax = plt.subplots(figsize=(10, 8.5)) # 稍微微調比例更適合放入論文排版
            
            # 計算該階段總 Dubins 飛行長度
            closed_idx = route.get_closed_path()
            total_len = 0.0
            for j in range(len(closed_idx) - 1):
                u, v = closed_idx[j], closed_idx[j+1]
                total_len += self.cost_calculator.cost(points[u], points[v])
                
            title = f"Stage {i}: {i} Dynamic Targets Added\nTotal Dubins Trajectory Length: {total_len:.2f} km"
            self._draw_state(ax, centers, points, route, start, title, phi, radii)
            
            save_path = f"{save_dir}/tspn_stage_{i}.png"
            # 同步將 facecolor 改為 white
            plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
            plt.close(fig)
            print(f"快照已保存: {save_path}")