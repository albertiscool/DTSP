import numpy as np
from core.routing import Route

class CheapestInsertion:
    def __init__(self, cost_fn):
        self.cost_fn = cost_fn

    def insert(self, route, new_point, points):
        # 支援傳入 Route 物件或純 index 列表
        is_route_obj = isinstance(route, Route)
        indices = list(route.indices) if is_route_obj else list(route)
        
        if not indices:
            # 若原路徑為空，預設新加入的點為唯一的起點
            return Route([0]) if is_route_obj else [0]

        best_pos = None
        best_cost = float('inf')
        
        # 新目標點在 points 陣列中的索引（假設呼叫前已先將點加入 points 尾端）
        new_node_idx = len(points) - 1

        for i in range(len(indices)):
            idx_a = indices[i]
            idx_b = indices[(i + 1) % len(indices)]
            
            a = points[idx_a]
            b = points[idx_b]

            cost_remove = self.cost_fn(a, b)
            cost_add = self.cost_fn(a, new_point) + self.cost_fn(new_point, b)
            delta = cost_add - cost_remove

            if delta < best_cost:
                best_cost = delta
                best_pos = i + 1

        indices.insert(best_pos, new_node_idx)
        return Route(indices) if is_route_obj else indices