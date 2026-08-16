# DTSP: Dynamic Traveling Salesman Problem & UAV Path Planning Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

本專案是一個針對 **動態旅行推銷員問題 (Dynamic Traveling Salesman Problem, DTSP)**、**帶鄰域與避障約束的 Dubins 航跡規劃 (DTSPN with Obstacle Avoidance)** 以及 **前沿深度強化學習混合求解 (HyDR-TR)** 的研究與實作代碼庫。

專案聚焦於無人機 (UAV) 在真實海洋/空域環境下的巡航任務：在考慮無人機**最小轉彎半徑 (Dubins Kinematics)**、**雷達動態發現新目標**、**5~9 km 環形觀測帶** 以及 **5 km 敵艦防空火網禁航區避障** 的複合約束下，實現全域與實時動態航跡規劃。

---

## 📑 目錄 (Table of Contents)
- [系統架構與模組概覽](#-系統架構與模組概覽)
- [核心演算法詳細解析](#-核心演算法詳細解析)
  - [1. 無人機動態航跡與避障規劃 (dtsp_uav)](#1-無人機動態航跡與避障規劃-dtsp_uav)
  - [2. 深度強化學習與樹搜尋混合框架 (HyDR-TR)](#2-深度強化學習與樹搜尋混合框架-hydr-tr)
- [數學模型與約束條件](#-數學模型與約束條件)
- [視覺化與快速上手指南](#-視覺化與快速上手指南)
- [目錄結構說明](#-目錄結構說明)
- [演算法優化與未來展望](#-演算法優化與未來展望)

---

## 🏛️ 系統架構與模組概覽

整個專案由兩大核心體系與一套基準數據集構成：

```
DTSP Project Root
├── dtsp_uav/        # 無人機物理動力學、DTSPN 避障與動態重規劃系統
├── HyDR-TR/         # GNN + 深度強化學習 + 樹搜尋 + GLKH 混合解算框架
├── TSPLIB/          # TSPLIB 國際標準 TSP 基準測試數據集
└── tspn_stage_*.png # 動態目標插入各階段的航跡演進快照圖
```

---

## 🔬 核心演算法詳細解析

### 1. 無人機動態航跡與避障規劃 (`dtsp_uav`)

針對具備非完整約束（Non-holonomic Constraints）的無人機，設計了兼顧即時性與全局避障的混合規劃演算法：

```
[環境目標點分佈]
      │
      ▼
[1. 幾何初解生成] ───> 最便宜插入啟發式 (Cheapest Insertion)
      │
      ▼
[2. Dubins 連續曲線] ──> CSC 航線計算 (LSL, RSR, LSR, RSL)
      │
      ▼
[3. 混合式模擬退火] ───> 同步優化【離散順序】+【連續航向與甜甜圈觀測點】
      │                  └── 空間離散取樣 + 禁航區侵入深度懲罰 (Penalty Function)
      ▼
[4. 事件觸發動態重規劃] ─> 雷達發現新敵艦時即時插入並局部細化
```

#### ① Dubins 物理航跡幾何演算法 (`core/dubins.py`)
* **物理模型**：無人機以固定航速 $v = 120\text{ km/h}$ 巡航，受限於轉彎半徑 $R = 2.0\text{ km}$，航跡由圓弧段 (C) 與直線段 (S) 組成。
* **CSC 模式求解**：實作了 4 種基本幾何模式：
  * **LSL** (Left - Straight - Left)
  * **RSR** (Right - Straight - Right)
  * **LSR** (Left - Straight - Right)
  * **RSL** (Right - Straight - Left)
  精確計算任意兩點位姿 $(x_1, y_1, \theta_1) \to (x_2, y_2, \theta_2)$ 間的最短 Dubins 曲線長度與插值軌跡。

#### ② 最便宜插入啟發式 (`algorithms/insertion.py`)
* **作用**：當雷達在巡航中動態發現第 $k$ 個新敵艦目標時，演算法遍歷現有閉環航線的所有邊 $(u, v)$：
  $$\Delta C = \text{Cost}(u, \text{new}) + \text{Cost}(\text{new}, v) - \text{Cost}(u, v)$$
  在 $O(N)$ 時間複雜度內將新目標插入到航程增量最小的位置，實現快速響應。

#### ③ 混合式模擬退火優化器 (`algorithms/simulated_annealing_tspn.py`)
同時優化 **離散變數** 與 **連續變數**：
1. **訪問序列 (離散)**：採用 2-opt 邊反轉擾動，解開全域交叉路徑。
2. **觀測方位角 $\phi_i \in [0, 2\pi)$ (連續)**：決定無人機切入觀測甜甜圈的角度。
3. **觀測半徑 $r_i \in [5\text{km}, 9\text{km}]$ (連續)**：自適應調整距目標的觀測距離。
4. **通過航向角 $\theta_i \in [0, 2\pi)$ (連續)**：無人機通過觀測點時的機頭指向。

#### ④ 幾何避障碰撞檢測與懲罰函數法
* **碰撞檢測**：沿 Dubins 軌跡進行等間距空間取樣，計算軌跡與所有敵艦中心的歐氏距離。
* **懲罰函數**：若軌跡侵入 5km 禁航半徑，計算侵入深度 $\text{depth} = 5.0 - \text{dist}$，並計入適應度函數：
  $$\text{Fitness} = \text{Total Dubins Length} + \sum_{\text{collisions}} (\text{Penalty}_{\text{base}} + \text{Penalty}_{\text{slope}} \times \text{depth})$$

#### ⑤ 自適應初始溫度估算 (Adaptive Initial Temperature)
在退火開始前執行 50 次微小擾動取樣，計算能量波動平均值 $\overline{\Delta E}$，根據目標接受率 $P_0 \approx 80\%$ 自動推導初始溫度：
$$T_0 = -\frac{\overline{\Delta E}}{\ln(P_0)}$$

---

### 2. 深度強化學習與樹搜尋混合框架 (`HyDR-TR`)

結合圖神經網絡的特徵表徵能力與組合優化求解器的精確搜尋能力：

| 核心技術 | 檔案位置 | 功能與原理 |
| :--- | :--- | :--- |
| **GIN (Graph Isomorphism Network)** | `HyDR-TR/gin.py` | **圖同構神經網絡**：提取大規模目標點的分佈特徵、距離矩陣與拓撲圖嵌入 (Graph Embeddings)。 |
| **DRL (Deep Reinforcement Learning)** | `HyDR-TR/train_HyDR_TR.py` | **強化學習策略網絡**：以最小化 Makespan / 總距離為獎勵函數，端到端學習構造路徑策略。 |
| **Tree-based Refinement (TR)** | `HyDR-TR/policy_HyDR_TR.py` | **樹搜尋修飾**：在神經網絡給出初解後，透過多分支樹狀搜尋進行局部探索與航向微調。 |
| **GLKH / LKH 求解器** | `HyDR-TR/GLKH` | **Lin-Kernighan-Helsgaun 求解器**：將連續航向離散化為 GTSP 問題，求出極限理論最優解。 |
| **K-Means 多機任務分群** | `HyDR-TR/analyze_*.py` | **多無人機 (Multi-UAV) 任務指派**：將龐大目標群聚類分配給多架無人機協同作業。 |

---

## 📐 數學模型與約束條件

1. **無人機運動學模型 (Dubins Model)**：
   $$\begin{cases}
   \dot{x} = v \cos\theta \\
   \dot{y} = v \sin\theta \\
   \dot{\theta} = u, \quad |u| \le \frac{v}{R}
   \end{cases}$$

2. **鄰域觀測約束 (Observation Annulus Constraint)**：
   對任意目標中心 $c_i$，無人機的拜訪點 $p_i$ 必須落在環形觀測帶內：
   $$r_{\min} \le \|p_i - c_i\|_2 \le r_{\max} \quad (5\text{ km} \le r \le 9\text{ km})$$

3. **防空火網禁航避障約束 (No-Fly Zone Obstacle Avoidance)**：
   無人機連續飛行軌跡 $p(t)$ 必須始終保持在所有敵艦禁航半徑之外：
   $$\|p(t) - c_j\|_2 \ge r_{\text{obs}} \quad \forall t, \forall j \quad (r_{\text{obs}} = 5\text{ km})$$

---

## 🚀 視覺化與快速上手指南

### 1. 安裝環境依賴

```bash
# 基礎依賴
pip install numpy matplotlib scipy

# 深度學習與強化學習依賴 (如需執行 HyDR-TR)
pip install torch torch-geometric PyYAML tqdm scikit-learn ortools
```

### 2. 執行無人機 DTSPN 動態避障巡航模擬

此腳本將模擬 25 個初始敵艦目標，並動態加入 5 個新目標，完成全流程航跡規劃並輸出 6 張高畫質快照：

```bash
python dtsp_uav/main_tspn.py
```
> 輸出圖片將保存為 `tspn_stage_0.png` 至 `tspn_stage_5.png`。

### 3. 執行基礎 Dubins DTSP 模擬

```bash
python dtsp_uav/main.py
```

### 4. 執行 GLKH 精確求解並繪製 TSPLIB 基準軌跡

```bash
cd HyDR-TR
python solve_single_dtsp_glkh.py
```

### 5. 視覺化 AI 模型多機路徑

```bash
cd HyDR-TR
python visualize_routes.py --file TSPLIB_30runs/berlin52_results.pkl --alg HyDR-TR --run 0
```

---

## 📂 目錄結構說明

```
.
├── README.md                  # 專案總覽與技術說明文件
├── problem.tsp                # TSPLIB 格式標準問題定義檔
├── tspn_stage_0.png ~ 5.png   # DTSPN 各階段動態重規劃可視化成果圖
│
├── dtsp_uav/                  # 無人機 DTSP/DTSPN 航跡規劃核心專案
│   ├── main.py                # 基礎 DTSP 主程式 (Cheapest Insertion + SA)
│   ├── main_tspn.py           # DTSPN 避障與鄰域觀測主程式
│   ├── solve_glkh.py          # 呼叫 GLKH 求解器之介面
│   ├── glkh_tspn.py           # GLKH DTSPN 比對程式
│   ├── core/                  # 核心物理與環境模型 (dubins, environment, routing)
│   ├── algorithms/            # 規劃演算法 (insertion, simulated_annealing_tspn)
│   ├── dynamic/               # 動態重規劃器 (replanner)
│   ├── simulation/            # 視覺化渲染器 (animator_tspn)
│   └── tsplib/                # TSPLIB 測試集檔案
│
└── HyDR-TR/                   # 深度強化學習與組合優化混合框架
    ├── train_HyDR_TR.py       # HyDR-TR 模型訓練腳本
    ├── train_GIN_GLKH.py      # GIN-GLKH 基準模型訓練腳本
    ├── policy_HyDR_TR.py      # HyDR-TR 策略神經網絡
    ├── gin.py                 # Graph Isomorphism Network 定義
    ├── analyze_*.py           # TSPLIB、敏感度與隨機數據集評估分析腳本
    ├── visualize_routes.py    # 路徑視覺化工具
    └── saved_model/           # 預訓練神經網絡權重
```

---

## 💡 演算法優化與未來展望

目前的模擬程式已成功驗證 **「Dubins 機動限制 + 5km 禁航區避障 + 5-9km 環形觀測帶」** 的混合退火優化機制。未來可進一步進行以下改良：

1. **凸包切線幾何初解**：利用 2D 凸包或最便宜插入法提供高品質初始 TSP 訪問環，避免退火演算法陷入局部交叉。
2. **切線航向連續性平滑 (Tangential Heading Smoothing)**：將各觀測點的通過航向角直接約束在相鄰點連線的切線方向，徹底消除無人機的大角度迴圈轉彎。
3. **DRL 與 DTSPN 結合**：將 `HyDR-TR` 的圖神經網絡引入 `dtsp_uav` 實現毫秒級神經網絡即時避障重規劃。
