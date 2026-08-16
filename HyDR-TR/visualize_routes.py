import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# Use a universally available fallback font
plt.rcParams['font.family'] = 'DejaVu Sans'

def plot_instance_routes(result_file, algorithm='HyDR-TR', run_idx=0, save_path=None):
    """
    從 pickle 結果檔案中視覺化特定運行次數的路徑。
    """
    if not os.path.exists(result_file):
        print(f"錯誤: 找不到檔案 {result_file}")
        return

    with open(result_file, 'rb') as f:
        results = pickle.load(f)

    # 檢查資料結構 (支援單一 Case 或多 Case 結構)
    if algorithm not in results:
        # 嘗試檢查是否為多 Case 結構中的某個實例
        print(f"錯誤: 在結果中找不到演算法 '{algorithm}'。可用鍵值: {list(results.keys())}")
        return

    alg_data = results[algorithm]
    
    if run_idx >= len(alg_data['routes']):
        print(f"錯誤: 指定的 run_idx {run_idx} 超出範圍 (總計 {len(alg_data['routes'])} 筆記錄)")
        return

    routes = alg_data['routes'][run_idx]
    makespan = alg_data['makespan'][run_idx]
    total_dist = alg_data['total_distance'][run_idx]
    seed = alg_data['seeds'][run_idx]

    plt.figure(figsize=(12, 10))
    # 定義顏色循環
    colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
    
    # 繪製路徑
    for i, (vehicle_id, path) in enumerate(routes.items()):
        # path 格式為 [N, 3] -> (x, y, yaw)
        plt.plot(path[:, 0], path[:, 1], color=colors[i], label=f'Vehicle {vehicle_id}', linewidth=2)
        
        # 繪製起點（倉儲/Depot）
        plt.scatter(path[0, 0], path[0, 1], c='black', marker='s', s=100, zorder=5)
        
        # 繪製方向箭頭（可選）
        idx = len(path) // 2
        plt.arrow(path[idx, 0], path[idx, 1], 
                  np.cos(path[idx, 2])*0.1, np.sin(path[idx, 2])*0.1,
                  head_width=5, color=colors[i], alpha=0.5)

    plt.title(f"Instance Routes - {algorithm}\nMakespan: {makespan:.2f} | Total: {total_dist:.2f} | Seed: {seed}", fontsize=14)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"結果已保存至: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='視覺化 DTSP 路徑結果')
    parser.add_argument('--file', type=str, required=True, help='結果 pickle 檔案路徑 (例如: TSPLIB_30runs/berlin52_results.pkl)')
    parser.add_argument('--alg', type=str, default='HyDR-TR', help='要顯示的演算法名稱 (HyDR-TR, KM-GLKH, GIN-GLKH)')
    parser.add_argument('--run', type=int, default=0, help='第幾次運行的索引 (預設: 0)')
    parser.add_argument('--save', type=str, default=None, help='保存圖片的路徑')
    
    args = parser.parse_args()
    
    # 確保字體支援中文 (視環境而定，此處使用預設)
    plt.rcParams['font.sans-serif'] = ['Arial'] 
    
    plot_instance_routes(args.file, args.alg, args.run, args.save)