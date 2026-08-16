import numpy as np
import os
import matplotlib.pyplot as plt

# 從 analyze_AA_GLKH_correlation.py 導入 calculate_glkh_cost 函數
# 由於 calculate_glkh_cost 依賴於 utils.glkh_dtsp 和 utils.dubins.dubins_length_only，
# 我們需要確保這些模組在執行時是可用的。
# 為了簡化，我們直接將其定義複製過來，避免複雜的模組導入路徑問題。
from utils.glkh_dtsp import glkh_dtsp_solve_originfixed
from utils.dubins.dubins_length_only import dubins_path_length
from utils.dubins.dubins_path import plan_dubins_path

def solve_and_visualize_glkh(instance, turning_radius, tsp_name="Instance"):
    """使用 GLKH 求解 DTSP 並根據要求視覺化路徑軌跡"""
    try:
        curvature = 1.0 / turning_radius  # Convert turning radius to curvature
        
        # Solve DTSP using GLKH with fixed origin heading
        route, route_headings = glkh_dtsp_solve_originfixed(instance, 8, curvature)
        
        # Make dubins waypoints
        dubins_waypoints = np.hstack((np.array(instance[route]), 
                                      np.array(route_headings).reshape(-1, 1)))
        
        # 用於存儲完整軌跡點
        all_path_x = []
        all_path_y = []
        total_length = 0

        for i in range(len(dubins_waypoints)):
            start_x = dubins_waypoints[i, 0]
            start_y = dubins_waypoints[i, 1]
            start_yaw = dubins_waypoints[i, 2]

            if i < len(dubins_waypoints) - 1:
                end_x = dubins_waypoints[i+1, 0]
                end_y = dubins_waypoints[i+1, 1]
                end_yaw = dubins_waypoints[i+1, 2]
            else:
                # Return to start for closed tour
                end_x = dubins_waypoints[0, 0]
                end_y = dubins_waypoints[0, 1]
                end_yaw = dubins_waypoints[0, 2]

            # 使用 plan_dubins_path 獲取具體座標點
            px, py, _, _, lengths = plan_dubins_path(start_x, start_y, start_yaw, 
                                                     end_x, end_y, end_yaw, curvature)
            all_path_x.extend(px)
            all_path_y.extend(py)
            total_length += sum(lengths)
            
        # --- 視覺化部分 ---
        plt.figure(figsize=(12, 10))
        
        # 1. 繪製 Dubins 路徑 (藍色實線)
        plt.plot(all_path_x, all_path_y, 'b-', linewidth=1.5, alpha=0.7, label=f'Dubins Path (Total: {total_length:.2f})')
        
        # 2. 標示靜態目標 (紅色實心圓點)
        # 我們假設 instance[0] 是起點，其餘是目標
        plt.scatter(instance[1:, 0], instance[1:, 1], c='red', marker='o', s=50, label='Static Targets', zorder=3)
        
        # 3. 標示起點/終點 (金邊金星)
        plt.scatter(instance[0, 0], instance[0, 1], c='gold', marker='*', s=300, 
                    edgecolors='orange', linewidths=1.5, label='Start/End Node (Depot)', zorder=5)

        plt.title(f"GLKH DTSP Visualization: {tsp_name}\nTurning Radius: {turning_radius}", fontsize=14)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal')
        
        # 保存圖片
        save_fn = f"{tsp_name}_glkh_result.png"
        plt.savefig(save_fn, dpi=300, bbox_inches='tight')
        print(f"視覺化圖表已保存至: {save_fn}")
        plt.show()

        return total_length
        
    except Exception as e:
        print(f"GLKH solver failed: {e}")
        return None

if __name__ == "__main__":
    tsp_file = "TSPLIB/ulysses22.tsp"
    turning_radius = 2  # 從 database 中獲取 ulysses22 的轉彎半徑

    # 讀取 TSPLIB 檔案
    coordinates = []
    with open(tsp_file, 'r') as f:
        for line in f:
            if line.strip() == 'NODE_COORD_SECTION':
                break
        for line in f:
            if line.strip() == 'EOF':
                break
            parts = line.split()
            if len(parts) == 3:
                coordinates.append((float(parts[1]), float(parts[2])))
    
    instance_data = np.array(coordinates)

    print(f"正在使用 GLKH 求解器計算 {tsp_file} 的 DTSP 成本...")
    cost = solve_and_visualize_glkh(instance_data, turning_radius, "ulysses22")

    if cost is not None:
        print(f"GLKH 求解器為 {tsp_file} 計算出的 DTSP 成本為: {cost:.4f}")
    else:
        print(f"未能為 {tsp_file} 計算 DTSP 成本。")