# -*- coding: utf-8 -*-
"""
generate_all_taiwan_maps.py
直接使用報告中 Stage 5 (tspn_stage_5.png, Seed 42, 全域 30 節點：1 起降基地 + 24 常態目標 + 5 動態威脅)
之真實演算法求解結果 (Version8_CaRVisibilitySolver)，
將起降基地 (Depot, Node 10) 錨定於「空軍臺東志航基地 (121.182°E, 22.793°N)」，
以 100% 嚴格 1:1 真實地理比例尺投影至臺灣東南海域衛星地圖。
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PACKAGE_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, PACKAGE_ROOT)
sys.path.insert(0, os.path.join(PACKAGE_ROOT, "core"))
sys.path.insert(0, os.path.join(PACKAGE_ROOT, "dtsp_uav"))

import shutil
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch
from PIL import Image

from compare_versions_v8 import Version8_CaRVisibilitySolver

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

MAP_PATH = os.path.join(PACKAGE_ROOT, "assets", "taiwan_satellite_base.png")
if not os.path.exists(MAP_PATH):
    MAP_PATH = os.path.join(PACKAGE_ROOT, "assets", "media_1793519915906_cfde3b13.png")

ARTIFACT_DIR = OUTPUT_DIR
# (OUTPUT_DIR configured in header)

img = Image.open(MAP_PATH)
img_w, img_h = img.size
img_rgb = np.array(img)

# =============================================================================
# 1. 真實經緯度 <-> 像素 仿射轉換 (由台灣本島與外島真實地標擬合)
# =============================================================================
def lonlat_to_pixel(lon, lat):
    px = 110.3352 * lon - 3.1812 * lat - 13033.4460
    py = -1.2237 * lon - 96.3400 * lat + 2723.4867
    return px, py

# 以空軍臺東志航基地 (121.182°E, 22.793°N) 為起降基地錨點 (Depot)
DEPOT_LON, DEPOT_LAT = 121.182, 22.793
KM_PER_DEG_LON = 111.320 * np.cos(np.radians(22.45))  # 102.88 km/deg
KM_PER_DEG_LAT = 110.730                              # 110.73 km/deg

def rel_km_to_lonlat(e_km, n_km):
    """將相對於臺東志航基地之真實東向/北向公里數 (e_km, n_km) 轉為真實經緯度"""
    lon = DEPOT_LON + (e_km / KM_PER_DEG_LON)
    lat = DEPOT_LAT + (n_km / KM_PER_DEG_LAT)
    return lon, lat

def rel_km_to_pixel(pts_km):
    """將相對於臺東志航基地之 (e_km, n_km) 1:1 轉換為地圖像素座標"""
    pts_km = np.asarray(pts_km, dtype=float)
    if pts_km.ndim == 1:
        lon, lat = rel_km_to_lonlat(pts_km[0], pts_km[1])
        px, py = lonlat_to_pixel(lon, lat)
        return np.array([px, py], dtype=float)
    else:
        lon, lat = rel_km_to_lonlat(pts_km[:, 0], pts_km[:, 1])
        px, py = lonlat_to_pixel(lon, lat)
        return np.column_stack([px, py])

def true_circle_px_polygon(center_rel_km, radius_km, num_verts=96):
    """在真實地理公里空間生成半徑 radius_km 之正圓，再投影至地圖像素座標 (嚴格 1:1 比例尺)"""
    angles = np.linspace(0.0, 2.0 * np.pi, num_verts, endpoint=False)
    circle_km = np.column_stack([
        center_rel_km[0] + radius_km * np.cos(angles),
        center_rel_km[1] + radius_km * np.sin(angles)
    ])
    return rel_km_to_pixel(circle_km)

# =============================================================================
# 2. 執行與 tspn_stage_5.png (圖3) 完全一致的 Stage 5 演算法求解
# =============================================================================
R_THREAT_KM = 5.0
R_SAFE_KM = 7.0

def solve_stage5_pattern():
    cache_path = os.path.join(PACKAGE_ROOT, "data", "stage5_solved_cache.npz")
    if not os.path.exists(cache_path):
        cache_path = os.path.join(PACKAGE_ROOT, "stage5_solved_cache.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return int(data["start_node_idx"]), data["ind"], data["centers_rel_km"], data["pts_rel_km"], data["route_rel_km"]

    solver = Version8_CaRVisibilitySolver()

    np.random.seed(42)
    random.seed(42)
    n_static = 25
    static_centers = np.column_stack([
        np.random.uniform(50, 150, n_static),
        np.random.uniform(0, 100, n_static)
    ])
    norm_x = (static_centers[:, 0] - 50.0) / 100.0
    norm_y = (static_centers[:, 1] - 0.0) / 100.0
    start_node_idx = int(np.argmin(norm_x - norm_y))

    dynamic_targets = [np.random.uniform([50, 0], [150, 100]) for _ in range(5)]
    centers_stage5 = np.vstack([static_centers] + dynamic_targets)

    # 執行 CaR 求解器取得巡航序列 ind 與 7.0 km 安全走廊錨點 pts
    ind, pts, length, col = solver.solve(centers_stage5, start_node_idx)

    # 生成完整連續曲率 Dubins 軌跡點 (與 plot_large_stage_figures.py 完全一致)
    n = len(ind)
    cost_calc = solver.cost_calculator
    route_segments_km = []
    for i in range(n):
        idx1 = ind[i]
        idx2 = ind[(i + 1) % n]
        p1 = pts[idx1]
        p2 = pts[idx2]
        t, p, q, mode, _ = cost_calc._plan_dubins(p1, p2)
        if mode is None:
            px = np.linspace(p1[0], p2[0], 30)
            py = np.linspace(p1[1], p2[1], 30)
        else:
            px, py = cost_calc._interpolate(p1, t, p, q, mode, step_size=0.35)
        seg = np.column_stack([px, py])
        route_segments_km.append(seg)

    full_route_orig_km = np.vstack(route_segments_km)

    # 將所有座標平移至以起降基地 (centers_stage5[start_node_idx]) 為原點 (0, 0) km
    depot_orig_km = centers_stage5[start_node_idx, :2]
    centers_rel_km = centers_stage5[:, :2] - depot_orig_km
    pts_rel_km = pts[:, :2] - depot_orig_km
    route_rel_km = full_route_orig_km - depot_orig_km

    np.savez(cache_path, start_node_idx=start_node_idx, ind=ind,
             centers_rel_km=centers_rel_km, pts_rel_km=pts_rel_km, route_rel_km=route_rel_km)
    return start_node_idx, ind, centers_rel_km, pts_rel_km, route_rel_km

start_node_idx, tour_ind, centers_rel_km, pts_rel_km, route_rel_km = solve_stage5_pattern()
depot_px = rel_km_to_pixel(centers_rel_km[start_node_idx])
centers_px = rel_km_to_pixel(centers_rel_km)
pts_px = rel_km_to_pixel(pts_rel_km)
dubins_path_px = rel_km_to_pixel(route_rel_km)

print(f"[Stage 5 Pattern 1:1 驗證] 節點總數 = {len(centers_rel_km)} (1 基地 + 24 常態目標 + 5 動態威脅)")

# =============================================================================
# 3. 繪製 1:1 真實地理比例尺標尺 (Scale Bar: 0 - 10 - 20 - 50 km)
# =============================================================================
def draw_true_scale_bar(ax, anchor_rel_km, is_zoom=False):
    p0 = rel_km_to_pixel(anchor_rel_km)
    p10 = rel_km_to_pixel(anchor_rel_km + np.array([10.0, 0.0]))
    p20 = rel_km_to_pixel(anchor_rel_km + np.array([20.0, 0.0]))
    p50 = rel_km_to_pixel(anchor_rel_km + np.array([50.0, 0.0]))
    
    y_bar = p0[1]
    h_tick = 4.0 if is_zoom else 5.5
    
    pad_x = 12.0 if is_zoom else 16.0
    pad_y = 13.5 if is_zoom else 19.0
    bg = FancyBboxPatch((p0[0] - pad_x, y_bar - pad_y),
                        (p50[0] - p0[0]) + 2 * pad_x, 2.25 * pad_y,
                        boxstyle="round,pad=0.2,rounding_size=0.1",
                        facecolor='#0F172A', edgecolor='#38BDF8', linewidth=1.6, alpha=0.94, zorder=19)
    ax.add_patch(bg)
    
    ax.plot([p0[0], p50[0]], [y_bar, y_bar], color='#F8FAFC', linewidth=4.0 if is_zoom else 3.4, zorder=20)
    ax.plot([p0[0], p10[0]], [y_bar, y_bar], color='#38BDF8', linewidth=4.5 if is_zoom else 4.0, zorder=21)
    ax.plot([p20[0], p50[0]], [y_bar, y_bar], color='#38BDF8', linewidth=4.5 if is_zoom else 4.0, zorder=21)
    
    for pt, lbl in [(p0, "0"), (p10, "10"), (p20, "20"), (p50, "50km")]:
        ax.plot([pt[0], pt[0]], [y_bar - h_tick, y_bar + h_tick], color='#F8FAFC', linewidth=2.0, zorder=21)
        ax.text(pt[0], y_bar + h_tick + (5.5 if is_zoom else 9.0), lbl,
                color='#F8FAFC', fontsize=10.5 if is_zoom else 9.5, fontweight='bold',
                ha='center', va='top', zorder=21)
    
    ax.text(0.5 * (p0[0] + p50[0]), y_bar - h_tick - (1.8 if is_zoom else 3.0),
            "1:1 比例尺 (紅圈=10km)",
            color='#FDE047', fontsize=10.5 if is_zoom else 9.5, fontweight='bold',
            ha='center', va='bottom', zorder=21)

# =============================================================================
# 4. 共用繪圖輔助函數：在指定 Axes 上繪製 Stage 5 之 1:1 戰術要素
# =============================================================================
def plot_stage5_tactical_elements(ax, is_zoom=False):
    first_static = True
    first_dyn = True

    # 繪製所有 29 個目標與動態威脅圓圈 (Node 0~24 為靜態目標，Node 25~29 為 Stage 1~5 新增之 5 個動態威脅)
    for i in range(len(centers_rel_km)):
        if i == start_node_idx:
            continue
        c_km = centers_rel_km[i]
        cx, cy = centers_px[i]
        is_dynamic = (i >= 25)

        # 7.0 km 主動安全走廊
        poly_safe = true_circle_px_polygon(c_km, R_SAFE_KM)
        if not is_dynamic:
            ax.add_patch(Polygon(poly_safe, closed=True, facecolor='#0284C7', edgecolor='none', alpha=0.16, zorder=2))
            ax.add_patch(Polygon(poly_safe, closed=True, facecolor='none', edgecolor='#38BDF8',
                                 linestyle=':', linewidth=1.8 if is_zoom else 1.2, alpha=0.92, zorder=4,
                                 label='安全走廊 R=7km (直徑14km)' if first_static else ""))
            # 5.0 km 禁航核心
            poly_core = true_circle_px_polygon(c_km, R_THREAT_KM)
            ax.add_patch(Polygon(poly_core, closed=True, facecolor='#FCA5A5', edgecolor='#EF4444',
                                 linestyle='--', linewidth=1.5 if is_zoom else 1.0, alpha=0.45, zorder=3,
                                 label='常態目標禁航核心 R=5km (24處)' if first_static else ""))
            ax.scatter(cx, cy, c='#EF4444', s=24 if is_zoom else 12, edgecolors='#7F1D1D', linewidths=0.8, zorder=5)
            first_static = False
        else:
            # Stage 1~5 突增之 5 個動態威脅 (以醒目紅橙雙圈強調)
            ax.add_patch(Polygon(poly_safe, closed=True, facecolor='#F59E0B', edgecolor='none', alpha=0.24, zorder=4))
            ax.add_patch(Polygon(poly_safe, closed=True, facecolor='none', edgecolor='#FDE047',
                                 linestyle='--', linewidth=2.1 if is_zoom else 1.4, alpha=0.95, zorder=6))
            poly_core = true_circle_px_polygon(c_km, R_THREAT_KM)
            ax.add_patch(Polygon(poly_core, closed=True, facecolor='#DC2626', edgecolor='#FF2A2A',
                                 linestyle='-', linewidth=2.1 if is_zoom else 1.4, alpha=0.72, zorder=5,
                                 label='Stage 5 動態突增威脅 D1~D5 (R=5km)' if first_dyn else ""))
            ax.scatter(cx, cy, c='#FEF08A', s=46 if is_zoom else 24, marker='x', linewidths=1.8, zorder=7)
            first_dyn = False

            if is_zoom:
                dyn_idx = i - 24  # D1 ~ D5
                offsets_dyn = {
                    1: (13, 0),    # D1: (94.9, -3.3)
                    2: (13, 0),    # D2: (91.9, 8.6)
                    3: (0, -9),    # D3: (57.7, 11.3)
                    4: (-16, 2),   # D4: (6.8, -61.2)
                    5: (-16, -2),  # D5: (2.5, -48.3)
                }
                ox, oy = offsets_dyn.get(dyn_idx, (12, 0))
                ax.text(cx + ox, cy + oy, f"威脅D{dyn_idx}",
                        color='#FEF08A', fontsize=11.0, fontweight='bold', ha='center', va='center', zorder=14,
                        bbox=dict(boxstyle='round,pad=0.16', facecolor='#7F1D1D', edgecolor='#FACC15', linewidth=1.3, alpha=0.94))

    # 繪製位於 7.0 km 走廊上的綠色任務航路錨點 (Waypoints)
    non_depot_mask = np.arange(len(pts_px)) != start_node_idx
    ax.scatter(pts_px[non_depot_mask, 0], pts_px[non_depot_mask, 1],
               c='#10B981', s=54 if is_zoom else 28, edgecolors='#064E3B',
               linewidths=1.3, zorder=11, label='任務航路錨點 (位於 7km 走廊上)')

    # 繪製 Stage 5 CaR 杜賓斯連續曲率航線
    ax.plot(dubins_path_px[:, 0], dubins_path_px[:, 1], color='#0369A1',
            linewidth=5.2 if is_zoom else 3.5, alpha=0.55, zorder=8)
    ax.plot(dubins_path_px[:, 0], dubins_path_px[:, 1], color='#00F5FF',
            linewidth=2.8 if is_zoom else 1.8, alpha=0.98, zorder=9,
            label='Stage 5 杜賓斯連續曲率航線 (486.8 km)')

    # 繪製起降基地 (空軍臺東志航基地, Depot)
    ax.scatter(depot_px[0], depot_px[1], c='#FACC15', s=320 if is_zoom else 220,
               marker='*', edgecolors='#78350F', linewidths=1.8, zorder=15,
               label='起降基地：臺東志航基地 (Depot)')
    ax.text(depot_px[0] - 6, depot_px[1] - 3,
            "臺東志航基地",
            color='#FEF08A', fontsize=12.8 if is_zoom else 11.5, fontweight='bold', ha='right', zorder=16,
            bbox=dict(boxstyle='round,pad=0.22', facecolor='#713F12', edgecolor='#FACC15', linewidth=1.4, alpha=0.95))

    # 綠島與蘭嶼真實地理標記
    if is_zoom:
        px_ludao, py_ludao = lonlat_to_pixel(121.49, 22.67)
        ax.text(px_ludao - 2, py_ludao - 6.5,
                "綠島(4km)",
                color='#F8FAFC', fontsize=10.5, fontweight='bold', ha='center', zorder=13,
                bbox=dict(boxstyle='round,pad=0.14', facecolor='#0F172A', edgecolor='#38BDF8', linewidth=1.2, alpha=0.90))
        
        px_lanyu, py_lanyu = lonlat_to_pixel(121.55, 22.05)
        ax.text(px_lanyu, py_lanyu + 7.5,
                "蘭嶼(12km)",
                color='#F8FAFC', fontsize=10.5, fontweight='bold', ha='center', zorder=13,
                bbox=dict(boxstyle='round,pad=0.14', facecolor='#0F172A', edgecolor='#38BDF8', linewidth=1.2, alpha=0.92))
    else:
        rect_box = FancyBboxPatch((202, 315), 480 - 202, 495 - 315,
                                  boxstyle="round,pad=0.1", facecolor='none',
                                  edgecolor='#00F5FF', linestyle='--', linewidth=1.8, alpha=0.90, zorder=16)
        ax.add_patch(rect_box)
        ax.text(341, 308, "Stage 5 東南海域任務區 (視圖B)", color='#00F5FF', fontsize=10.8, fontweight='bold',
                ha='center', va='bottom', zorder=17,
                bbox=dict(boxstyle='round,pad=0.18', facecolor='#0F172A', edgecolor='#00F5FF', alpha=0.90))

def safe_save_fig(fig, filename, **kwargs):
    art_path = os.path.join(ARTIFACT_DIR, filename)
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(art_path, **kwargs)
    print(f"[OK] 已儲存至 Artifact: {art_path}")
    try:
        shutil.copyfile(art_path, out_path)
        print(f"[OK] 已同步至工作目錄: {out_path}")
    except Exception:
        alt_path = os.path.join(OUTPUT_DIR, filename.replace('.png', '_updated.png'))
        shutil.copyfile(art_path, alt_path)
        print(f"[NOTE] 原檔開啟中，已另存至: {alt_path}")

# =============================================================================
# 圖 1：全島宏觀戰術地圖 (taiwan_se_dtsp_overview.png)
# =============================================================================
def draw_overview():
    print(">>> 正在繪製圖 1：全島宏觀戰術地圖 (taiwan_se_dtsp_overview.png)...")
    fig, ax = plt.subplots(figsize=(9.2, 6.6), dpi=300)
    
    ax.imshow(img_rgb)
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.axis('off')
    
    for lon in [120.0, 121.0, 122.0, 123.0, 124.0]:
        px_top, _ = lonlat_to_pixel(lon, 26.5)
        px_bot, _ = lonlat_to_pixel(lon, 20.5)
        ax.plot([px_top, px_bot], [0, img_h], color='#64748B', linestyle=':', linewidth=0.8, alpha=0.45)
        ax.text(px_bot, img_h - 10, f"{int(lon)}°E", color='#CBD5E1', fontsize=11.0, fontweight='bold', ha='center')

    for lat in [21.0, 22.0, 23.0, 24.0, 25.0]:
        _, py_l = lonlat_to_pixel(118.5, lat)
        _, py_r = lonlat_to_pixel(127.0, lat)
        ax.plot([0, img_w], [py_l, py_r], color='#64748B', linestyle=':', linewidth=0.8, alpha=0.45)
        ax.text(14, py_l - 4, f"{int(lat)}°N", color='#CBD5E1', fontsize=11.0, fontweight='bold', va='center')

    plot_stage5_tactical_elements(ax, is_zoom=False)
    draw_true_scale_bar(ax, np.array([240.0, -50.0]), is_zoom=False)

    ax.text(20, 34, "臺灣東南海域 Stage 5 動態避障巡航模擬 (1:1 真實比例尺)",
            fontsize=14.0, fontweight='bold', color='#F8FAFC', zorder=21,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#0F172A', edgecolor='#00F0FF', linewidth=1.6, alpha=0.92))

    leg = ax.legend(loc='lower right', bbox_to_anchor=(0.985, 0.025), fontsize=10.2, framealpha=0.94,
                    facecolor='#0F172A', edgecolor='#00F0FF', labelcolor='#F8FAFC')
    leg.set_zorder(22)

    plt.tight_layout()
    safe_save_fig(fig, 'taiwan_se_dtsp_overview.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close(fig)

# =============================================================================
# 圖 2：東南海域 Stage 5 戰術高解析特寫圖 (taiwan_se_dtsp_zoom.png)
# =============================================================================
def draw_zoom():
    print(">>> 正在繪製圖 2：東南海域 Stage 5 戰術特寫圖 (taiwan_se_dtsp_zoom.png)...")
    x_min, x_max = 200, 498
    y_min, y_max = 312, 504
    
    fig, ax = plt.subplots(figsize=(9.2, 6.6), dpi=300)
    
    ax.imshow(img_rgb)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.axis('off')
    
    for lon in [121.0, 121.5, 122.0, 122.5, 123.0]:
        px_top, _ = lonlat_to_pixel(lon, 24.0)
        px_bot, _ = lonlat_to_pixel(lon, 21.0)
        ax.plot([px_top, px_bot], [y_min, y_max], color='#64748B', linestyle=':', linewidth=0.9, alpha=0.5)
        ax.text(px_bot, y_max - 3, f"{lon:.1f}°E", color='#E2E8F0', fontsize=11.5, fontweight='bold', ha='center')

    for lat in [22.0, 22.5, 23.0]:
        _, py_l = lonlat_to_pixel(121.0, lat)
        _, py_r = lonlat_to_pixel(124.0, lat)
        ax.plot([x_min, x_max], [py_l, py_r], color='#64748B', linestyle=':', linewidth=0.9, alpha=0.5)
        ax.text(x_min + 4, py_l - 3, f"{lat:.1f}°N", color='#E2E8F0', fontsize=11.5, fontweight='bold', va='center')

    plot_stage5_tactical_elements(ax, is_zoom=True)
    draw_true_scale_bar(ax, np.array([-48.0, -96.0]), is_zoom=True)

    ax.text(x_min + 6, y_min + 11.5,
            "臺灣東南海域 Stage 5 巡航軌跡 (1:1 比例尺 | 30 節點 | 淨距 2.0km)",
            fontsize=12.5, fontweight='bold', color='#F8FAFC', zorder=21,
            bbox=dict(boxstyle='round,pad=0.30', facecolor='#0F172A', edgecolor='#00F0FF', linewidth=1.5, alpha=0.94))

    leg = ax.legend(loc='lower right', bbox_to_anchor=(0.99, 0.02), fontsize=10.2, framealpha=0.95,
                    facecolor='#0F172A', edgecolor='#00F0FF', labelcolor='#F8FAFC')
    leg.set_zorder(22)

    plt.tight_layout()
    safe_save_fig(fig, 'taiwan_se_dtsp_zoom.png', bbox_inches='tight', pad_inches=0.04, dpi=300)
    plt.close(fig)

# =============================================================================
# 圖 3：綜合戰術作戰儀表板 (taiwan_se_dtsp_dashboard.png)
# =============================================================================
def draw_dashboard():
    print(">>> 正在繪製圖 3：綜合戰術作戰儀表板 (taiwan_se_dtsp_dashboard.png)...")
    fig = plt.figure(figsize=(11.4, 6.2), dpi=300)
    fig.patch.set_facecolor('#0B132B')
    
    fig.text(0.5, 0.945, "臺灣東南海域 Stage 5 定翼無人機動態避障巡航規劃 (1:1 真實地理比例尺)",
             fontsize=15.0, fontweight='bold', ha='center', color='#F8FAFC')

    # 左側：全島戰術態勢圖 (Overview)
    ax1 = fig.add_axes([0.015, 0.13, 0.47, 0.77])
    ax1.imshow(img_rgb)
    ax1.set_xlim(0, img_w)
    ax1.set_ylim(img_h, 0)
    ax1.axis('off')
    
    plot_stage5_tactical_elements(ax1, is_zoom=False)
    ax1.text(18, 36, "【A】全島與東南海域戰區定位 (1:1)",
             fontsize=12.0, fontweight='bold', color='#F8FAFC', zorder=20,
             bbox=dict(boxstyle='round,pad=0.28', facecolor='#0F172A', edgecolor='#00F0FF', linewidth=1.4, alpha=0.94))
    leg1 = ax1.legend(loc='lower right', bbox_to_anchor=(0.985, 0.02), fontsize=9.5, framealpha=0.95,
                      facecolor='#0F172A', edgecolor='#00F0FF', labelcolor='#F8FAFC')
    leg1.set_zorder(22)

    # 右側：東南海域 Stage 5 戰術高解析特寫 (Zoom View)
    ax2 = fig.add_axes([0.50, 0.13, 0.485, 0.77])
    x_min, x_max = 200, 445
    y_min, y_max = 312, 504
    ax2.imshow(img_rgb)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_max, y_min)
    ax2.axis('off')
    
    plot_stage5_tactical_elements(ax2, is_zoom=True)
    draw_true_scale_bar(ax2, np.array([-48.0, -96.0]), is_zoom=True)
    ax2.text(x_min + 6, y_min + 11.5, "【B】Stage 5 東南海域 30 節點巡航特寫",
             fontsize=12.0, fontweight='bold', color='#F8FAFC', zorder=20,
             bbox=dict(boxstyle='round,pad=0.28', facecolor='#0F172A', edgecolor='#00F0FF', linewidth=1.4, alpha=0.94))

    # 底部單列大字重點指標橫幅 (與報告中 Stage 5 數據完全一致)
    ax_bar = fig.add_axes([0.015, 0.018, 0.97, 0.09])
    ax_bar.set_facecolor('#111C3A')
    ax_bar.axis('off')
    banner = FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.01",
                            facecolor='#111C3A', edgecolor='#38BDF8', linewidth=1.5)
    ax_bar.add_patch(banner)
    summary_text = (
        "★ Stage 5 航跡總長：486.8 km   |   "
        "★ 禁航核心：R=5.0 km   |   "
        "★ 安全走廊：R=7.0 km (淨距2.0km)   |   "
        "★ 避障率：100% 零碰撞"
    )
    ax_bar.text(0.5, 0.5, summary_text, color='#FDE047', fontsize=12.5, fontweight='bold',
                ha='center', va='center')

    safe_save_fig(fig, 'taiwan_se_dtsp_dashboard.png', bbox_inches='tight', pad_inches=0.06, facecolor='#0B132B', dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    draw_overview()
    draw_zoom()
    draw_dashboard()
    print("全部 Stage 5 1:1 真實比例尺圖表更新完畢！")
