import numpy as np

class Environment:
    def __init__(self, x_range=(50, 150), y_range=(0, 100), n_static=25):
        self.x_range = x_range
        self.y_range = y_range
        self.cruise_speed = 120.0  # km/h
        self.targets = self._generate_targets(n_static)
        self.dynamic_targets = []
        self.clusters = self._generate_clusters(n_static)  # 新增群信息

    def _generate_targets(self, n):
        # 在指定的離岸 50km-150km, 南北 100km 範圍內生成
        x = np.random.uniform(self.x_range[0], self.x_range[1], size=n)
        y = np.random.uniform(self.y_range[0], self.y_range[1], size=n)
        return np.column_stack((x, y))

    def _generate_clusters(self, n):
        # 假設最多有4個群
        clusters = np.random.randint(1, 5, size=n)
        return clusters

    def add_dynamic_target(self, point):
        self.dynamic_targets.append(np.array(point))
        # 新增動態目標點的分群信息
        self.clusters = np.concatenate([self.clusters, np.random.randint(1, 5, size=1)])

    def get_all_targets(self, return_clusters=False):
        if not self.dynamic_targets:
            targets = self.targets
        else:
            targets = np.vstack([self.targets, np.array(self.dynamic_targets)])
        if return_clusters:
            return targets, self.clusters
        return targets

    def get_start_node_index(self):
        """找出搜尋區域內最左上方（最小x且最大y）的目標點作為起點"""
        all_targets = self.get_all_targets()
        # 尋找距離理論左上角點 (x_min, y_max) 最近的目標
        top_left_ref = np.array([self.x_range[0], self.y_range[1]])
        distances = np.linalg.norm(all_targets - top_left_ref, axis=1)
        return np.argmin(distances)