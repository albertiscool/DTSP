import numpy as np

class Route:
    def __init__(self, indices=None):
        """
        儲存目標點的訪問順序索引。
        :param indices: 目標點索引列表，例如 [0, 5, 2, ...]
        """
        self.indices = list(indices) if indices is not None else []

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.indices[item]

    def get_closed_path(self):
        """回傳包含回到起點的完整索引序列 (Closed Loop)"""
        if not self.indices:
            return []
        return self.indices + [self.indices[0]]

    def get_coordinates(self, points):
        """根據給定的點位陣列，回傳此路徑對應的座標序列"""
        return points[self.indices]

    def calculate_total_cost(self, points, cost_fn):
        """
        計算整條閉環路徑的總成本 (Dubins 距離)。
        """
        if len(self.indices) < 2:
            return 0.0
        
        total = 0.0
        closed_idx = self.get_closed_path()
        for i in range(len(closed_idx) - 1):
            u, v = closed_idx[i], closed_idx[i+1]
            total += cost_fn(points[u], points[v])
        return total

    def __repr__(self):
        return f"Route({self.indices})"