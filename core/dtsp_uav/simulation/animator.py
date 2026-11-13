import matplotlib.pyplot as plt
import numpy as np
from core.dubins import DubinsCost

class Animator:
    def __init__(self, env):
        """
        初始化模擬器的視覺化介面。
        :param env: Environment 實例，包含空域範圍資訊
        """
        self.env = env
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.turning_radius = 1.0  # 預設轉彎半徑，應與 DubinsCost 一致
        self.n_static = len(env.targets)
        self.cost_calculator = DubinsCost(self.turning_radius)

    def _setup_axes(self, ax):
        """設定座標軸與範圍"""
        ax.set_xlim(self.env.x_range[0] - 10, self.env.x_range[1] + 10)
        ax.set_ylim(self.env.y_range[0] - 10, self.env.y_range[1] + 10)
        ax.set_aspect('equal')
        ax.set_xlabel("East (km)")
        ax.set_ylabel("North (km)")
        ax.grid(True, linestyle='--', alpha=0.5)

    def _interpolate_dubins_path(self, p1, p2):
        """
        呼叫 Dubins 邏輯產生插值點。
        註：目前的 get_path_points 實作為直線簡化版，
        若需要完美曲線，可在 get_path_points 中補全三角函數插值。
        """
        self.cost_calculator.R = self.turning_radius
        path_x, path_y = self.cost_calculator.get_path_points(p1, p2)
        return path_x, path_y

    def _draw_state(self, ax, points, route, start_node_idx, title):
        """核心繪圖邏輯，可在指定 axis 上繪製"""
        ax.clear()
        self._setup_axes(ax)
        ax.set_title(title)

        # 1. 分類繪製目標點
        # 靜態目標 (紅色)
        ax.scatter(points[:self.n_static, 0], points[:self.n_static, 1], 
                   c='red', s=25, label='Static Targets', zorder=5)
        
        # 動態目標 (青色)
        if len(points) > self.n_static:
            ax.scatter(points[self.n_static:, 0], points[self.n_static:, 1], 
                       c='cyan', s=50, edgecolors='blue', label='Dynamic Targets', zorder=5)
            # 明顯標示最新增加的點 (亮綠色大星號)
            ax.scatter(points[-1, 0], points[-1, 1], 
                       c='lime', s=200, marker='*', edgecolors='black', label='Newest Target', zorder=8)
        
        # 2. 標記起點/終點 (左上角目標)
        ax.scatter(points[start_node_idx, 0], points[start_node_idx, 1], 
                        c='gold', s=100, marker='*', edgecolors='black', label='Start', zorder=6)

        # 3. 繪製 Dubins 路徑
        closed_idx = route.get_closed_path()
        if len(closed_idx) > 1:
            for i in range(len(closed_idx) - 1):
                px, py = self._interpolate_dubins_path(points[closed_idx[i]], points[closed_idx[i+1]])
                ax.plot(px, py, 'b-', alpha=0.7, linewidth=1.0)

        # 4. 繪製航向箭頭
        if points.shape[1] == 3:
            u = np.cos(points[:, 2])
            v = np.sin(points[:, 2])
            ax.quiver(points[:, 0], points[:, 1], u, v, color='green', 
                           pivot='middle', width=0.005, headwidth=4, alpha=0.8, zorder=7)
        # 5. 顯示圖例
        ax.legend(loc='upper right', fontsize='small')

    def render(self, points, route, start_node_idx):
        """
        繪製當前狀態。
        """
        self._draw_state(self.ax, points, route, start_node_idx, "DTSP Real-time Update")
        plt.draw()
        plt.pause(0.1)

    def render_snapshots(self, snapshots):
        """
        跳出 6 張獨立的圖表展示不同階段
        :param snapshots: 包含 (points, route, start_idx) 的列表
        """
        plt.close(self.fig)  # 關閉原本的互動視窗
        
        for i, (p, r, start) in enumerate(snapshots):
            fig, ax = plt.subplots(figsize=(10, 8))
            # 計算當前快照的總成本
            cost = r.calculate_total_cost(p, self.cost_calculator.cost)
            title = f"Stage {i}: {i} Dynamic Targets Added\nTotal Distance: {cost:.2f} km"
            self._draw_state(ax, p, r, start, title)
        
        plt.show()

    def show(self):
        plt.show()