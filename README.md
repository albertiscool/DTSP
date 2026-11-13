# 無人機動態威脅巡航路徑規劃系統 (DTSP / CaR-Visibility Framework)
> **UAV Dynamic Threat Cruise Path Planning & Multi-Objective Mission System**  
> 國立中正大學 電腦工程學系 智慧系統實驗室

---

## 📖 專案簡介 (Overview)

本專案針對**固定翼無人機（Fixed-Wing UAV）**在廣達 $100\text{ km} \times 100\text{ km}$ 的非對稱對抗戰術空域中，執行多目標情監偵巡航與突發防空威脅動態避障任務，提出兼具**神經拓撲極速泛化**與**解析幾何絕對安全保證**的**「構建與修復（Construct-and-Refine; CaR）」雙層解耦動態路徑規劃架構**：

* **第一層（宏觀拓撲構建）**：基於多頭自注意力機制之 **Transformer 指針網路（Pointer Network）**，於 **0.06 秒** 內輸出全域巡航目標訪問順序骨架。
* **第二層（微觀幾何修復）**：基於解析幾何之 **可見性走廊凸外切線修復算子** 與 **Dubins 運動學連續曲率（$C^1$ 連續）平滑器**，顯式確立 $R_{\text{safe}} = 7.0\text{ km}$ 主動安全防護走廊（即威脅殺傷半徑 $5.0\text{ km}$ 外加 $2.0\text{ km}$ 實體安全防護淨距），達成 **100% 零碰撞避障**。
* **實境空域驗證**：以 **1:1 真實地理比例尺** 部署於臺灣東南海域（臺東志航基地、綠島、蘭嶼及巴士海峽外海），完成戰術情監偵實境投影。

---

## 📂 檔案目錄架構 (Directory Structure)

```text
DTSP_Package/
│
├── main.py                     # 一鍵執行主程式 (支援互動式選單與命令列參數)
├── requirements.txt            # Python 相依環境套件清單
├── README.md                   # 本使用說明文件
│
├── core/                       # 核心演算法模組庫
│   ├── compare_versions_v8.py  # CaR-Visibility 核心求解器 (Version 8)
│   ├── compare_versions_v5.py  # 基礎運動學、碰撞干涉檢測與幾何算子底層
│   ├── train_transformer_dtsp.py # Transformer 模型定義與強化學習訓練代碼
│   └── dtsp_uav/               # 基礎 Dubins 航跡推算與航線封裝庫
│
├── transformer_checkpoints/    # 預訓練神經網路權重
│   └── best_model.pt           # 訓練完成之 Transformer 模型權重 (8.9 MB)
│
├── data/                       # 測試資料庫與實驗數值數據
│   ├── stage5_solved_cache.npz # Stage 5 最佳化軌跡快取檔 (支援極速再現)
│   ├── three_methods_benchmark_results.csv # 三大方法 (SA vs LKH vs CaR) 原始指標數據
│   ├── unseen_51_100_results.csv # 50 組未知隨機地圖盲測數據集 (Seeds 51-100)
│   ├── unseen_50_results.csv   # 50 組未知隨機地圖評估數據
│   ├── oneshot_comparison_results.csv # 50 組基準與走廊過濾架構對比數據
│   └── monte_carlo_*.csv       # 蒙地卡羅多階段模擬測試數據集
│
├── assets/                     # 基礎底圖與高解析度圖表素材
│   ├── taiwan_satellite_base.png # 臺灣東南部海域高解析度衛星空照底圖
│   └── *.png                   # 9 張技術報告核心高對比圖表素材
│
├── docs/                       # 技術成果報告文件
│   ├── 無人機動態威脅巡航路徑規劃.docx
│
├── scripts/                    # 獨立功能測試與視覺化腳本
│   ├── 1_run_stage_benchmarks.py    # 執行 Stage 0 ~ Stage 5 突發威脅演進模擬
│   ├── 2_run_taiwan_simulation.py   # 生成 臺灣東南海域 1:1 實境戰術軌跡圖
│   ├── 3_run_unseen_benchmark.py    # 執行 50 組隨機未知地圖評估與幾何診斷
│   ├── 4_plot_comparison_metrics.py # 繪製三大演算法 (CaR vs LKH vs SA) 對比圖
│   └── 5_build_word_report.py       # 自動編譯產出 25 頁 Word 完整技術報告書
│
└── output/                     # 程式執行產出目錄 (所有新生成的圖表與報告均存放於此)
```

---

## 🛠️ 環境建置 (Installation)

### 1. 建立虛擬環境 (推薦使用 Conda 或 venv)
```bash
# 使用 Conda
conda create -n dtsp_env python=3.12 -y
conda activate dtsp_env

# 或使用 Python venv
python -m venv dtsp_env
# Windows 啟動:
dtsp_env\Scripts\activate
# Linux/macOS 啟動:
source dtsp_env/bin/activate
```

### 2. 安裝必要套件
```bash
pip install -r requirements.txt
```
*(若具備 NVIDIA 獨立顯示卡，可依照個人 CUDA 版本安裝對應之 PyTorch，如 `pip install torch --index-url https://download.pytorch.org/whl/cu118`)*

---

## 🚀 快速開始 (Quickstart)

### 方法一：一鍵控制台（互動式選單）
直接執行根目錄下的 `main.py`，依照終端機提示輸入代碼即可：
```bash
python main.py
```
```text
======================================================================
   無人機動態威脅巡航路徑規劃系統 (DTSP / CaR-Visibility Solver)
   UAV Dynamic Threat Cruise Path Planning Master Console
======================================================================
  請選擇欲執行的任務：
  [1] 執行 Stage 0 ~ Stage 5 突發防空威脅演進模擬 (生成各階段航跡圖)
  [2] 執行 臺灣東南海域 1:1 真實地理比例尺實境戰術圖 (志航基地/綠島/蘭嶼)
  [3] 執行 50 組隨機未知地圖 (Seeds 51-100) 泛化與幾何邊界診斷
  [4] 繪製 三大演算法 (CaR vs LKH vs SA) 核心指標與盒鬚圖
  [5] 一鍵編譯產出 25 頁 Word 完整技術報告書 (.docx)
  --------------------------------------------------------------------
  [A] 依序執行全部任務 (Run All Demos & Build Report)
  [0] 離開系統 (Exit)
======================================================================
```

### 方法二：命令列直接執行全部任務
```bash
python main.py --all
```

### 方法三：單獨執行特定任務
```bash
python scripts/1_run_stage_benchmarks.py   # 生成 Stage 0 ~ 5 航跡圖
python scripts/2_run_taiwan_simulation.py    # 生成 臺灣東南海域 1:1 戰術圖
python scripts/5_build_word_report.py        # 編譯產生最新技術報告 (.docx)
```

---

## 🎯 核心模擬腳本說明

| 腳本名稱 | 說明與產出內容 | 輸出檔案 (存於 `output/`) |
| :--- | :--- | :--- |
| `1_run_stage_benchmarks.py` | 模擬突發雷達防空威脅由 0 處逐步增至 5 處（Stage 0 至 5），展現線上動態切線避障與連續曲率軌跡平滑效果。 | `tspn_stage_0.png`<br>`tspn_stage_1.png`<br>`tspn_stage_5.png` |
| `2_run_taiwan_simulation.py` | 將 Stage 5 航路點以 1:1 真實地理比例尺投影至臺灣東南海域，以空軍臺東志航基地為起降點，包含綠島與蘭嶼實體比例尺對照。 | `taiwan_se_dtsp_overview.png`<br>`taiwan_se_dtsp_zoom.png`<br>`taiwan_se_dtsp_dashboard.png` |
| `3_run_unseen_benchmark.py` | 評估 50 組全新隨機未知地圖（Seeds 51–100），可視化雙威脅間距小於 3.0 km 的極端狹縫阻塞可行性幾何邊界。 | `unseen_difficult_seeds_trajectories.png` |
| `4_plot_comparison_metrics.py` | 產出與傳統模擬退火（Pure SA）、國際頂尖求解器 LKH 的核心航程、計算耗時、防護淨距對比長條圖與箱型圖。 | `sa_lkh_car_trajectories.png`<br>`sa_lkh_car_metrics_bar.png`<br>`sa_lkh_car_boxplots.png` |
| `5_build_word_report.py` | 自動調用 `python-docx` 整合所有圖表、數據表與學術論述，自動排版產出精準控制在 25 頁左右的專業技術成果報告書。 | `無人機動態威脅巡航路徑規劃_技術報告.docx` |

---

## 📊 核心任務參數對照 (Mission Parameters)

* **空域範圍**：$100\text{ km} \times 100\text{ km}$（臺灣東南海域離岸向東 50 至 150 km，南北 100 km）。
* **目標數量**：常態偵巡目標 20～25 個 + 動態突發防空威脅 3～5 個。
* **無人機動力學基準**：
  * 巡航前進速度 $v = 200.0\text{ m/s}$（約 $720\text{ km/h}$，對應高速定翼載台「勁蜂 IV 型」）。
  * 最小轉彎半徑 $\rho_{\text{min}} = 2.0\text{ km}$（高 G 機動可達 $1.5\text{ km}$）。
* **感測與安全防護指標**：
  * **ESM / ELINT 被動偵搜**：$50 \sim 100\text{ km}$ 遠距截獲敵方雷達輻射源與定位。
  * **雷達預警接收機 (RWR)**：$10 \sim 30\text{ km}$ 近距離防空鎖定告警，觸發 6.8 秒線上增量重規劃。
  * **機載海搜雷達 / 光電 (EO/IR)**：雷達探測距離 $80\text{ km}$，光電有效偵照極限 $10\text{ km}$。
  * **目標水平觀測距離**：$d_{\text{obs}} = 5.0 \sim 9.0\text{ km}$（避開 $5.0\text{ km}$ 內近程防空殺傷圈，且在 $10\text{ km}$ 光學可清晰識別視距內）。
  * **目標盤旋留空偵蒐時間**：$T_{\text{obs}} \ge 4.36\text{ 分鐘}$。
  * **主動安全防護淨距**：$D_{\text{clear}} = 2.00\text{ km}$（$R_{\text{safe}} = 7.0\text{ km}$ 與 $R_{\text{threat}} = 5.0\text{ km}$ 之緩衝差，100% 零碰撞保證）。
