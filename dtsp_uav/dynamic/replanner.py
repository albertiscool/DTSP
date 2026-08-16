import numpy as np

class Replanner:
    def __init__(self, insertion_algo):
        """
        即時重新規劃器。
        :param insertion_algo: 插入演算法實例 (例如 CheapestInsertion)
        """
        self.insertion_algo = insertion_algo

    def update(self, route, new_target, points):
        """
        當偵測到新目標時更新路徑。
        :param route: 目前的 Route 物件或索引列表
        :param new_target: 新目標的座標 [x, y] 或 [x, y, heading]
        :param points: 目前所有目標點的座標矩陣
        :return: (更新後的 Route 物件, 更新後的點位矩陣)
        """
        updated_points = np.vstack([points, new_target])
        # 呼叫插入演算法，它會根據 updated_points 的最後一個索引來更新 route
        updated_route = self.insertion_algo.insert(route, new_target, updated_points)
        return updated_route, updated_points