# -*- coding: utf-8 -*-
"""
=============================================================================
無人機動態威脅巡航路徑規劃系統 (DTSP / CaR-Visibility Framework)
UAV Dynamic Threat Cruise Path Planning & Multi-Objective Mission System
=============================================================================
主程式執行入口 (Main Entry Point)
可透過選單或命令列參數快速調用核心模擬與報告生成模組。
=============================================================================
"""

import os
import sys
import argparse
import subprocess

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PACKAGE_ROOT, "scripts")

def run_script(script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"[錯誤] 找不到腳本: {script_path}")
        return False
    print(f"\n========================================================")
    print(f">>> 正在執行: {script_name} ...")
    print(f"========================================================")
    cmd = [sys.executable, script_path]
    res = subprocess.run(cmd, cwd=PACKAGE_ROOT)
    if res.returncode == 0:
        print(f"[成功] {script_name} 執行完成！")
        return True
    else:
        print(f"[警告] {script_name} 執行過程中出現錯誤 (代碼: {res.returncode})")
        return False

def show_menu():
    print("""
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
""")
    choice = input("請輸入選項 (0/1/2/3/4/5/A): ").strip().upper()
    return choice

def main():
    parser = argparse.ArgumentParser(description="DTSP 無人機路徑規劃主程式")
    parser.add_argument("--stage", action="store_true", help="執行 Stage 0 ~ 5 模擬")
    parser.add_argument("--taiwan", action="store_true", help="執行 臺灣東南海域 1:1 實境戰術模擬")
    parser.add_argument("--unseen", action="store_true", help="執行 50 組隨機地圖診斷")
    parser.add_argument("--compare", action="store_true", help="繪製三大演算法效益對比圖")
    parser.add_argument("--report", action="store_true", help="編譯 25 頁 Word 技術報告")
    parser.add_argument("--all", action="store_true", help="依序執行全部任務")

    args = parser.parse_args()

    executed_any = False

    if args.all or args.stage:
        run_script("1_run_stage_benchmarks.py")
        executed_any = True

    if args.all or args.taiwan:
        run_script("2_run_taiwan_simulation.py")
        executed_any = True

    if args.all or args.unseen:
        run_script("3_run_unseen_benchmark.py")
        executed_any = True

    if args.all or args.compare:
        run_script("4_plot_comparison_metrics.py")
        executed_any = True

    if args.all or args.report:
        run_script("5_build_word_report.py")
        executed_any = True

    if executed_any:
        print("\n[完成] 指定命令列任務均已執行完畢！所有圖表與報告已儲存於 output/ 目錄。")
        return

    # 若未帶參數則進入互動選單
    while True:
        choice = show_menu()
        if choice == "1":
            run_script("1_run_stage_benchmarks.py")
        elif choice == "2":
            run_script("2_run_taiwan_simulation.py")
        elif choice == "3":
            run_script("3_run_unseen_benchmark.py")
        elif choice == "4":
            run_script("4_plot_comparison_metrics.py")
        elif choice == "5":
            run_script("5_build_word_report.py")
        elif choice == "A":
            run_script("1_run_stage_benchmarks.py")
            run_script("2_run_taiwan_simulation.py")
            run_script("3_run_unseen_benchmark.py")
            run_script("4_plot_comparison_metrics.py")
            run_script("5_build_word_report.py")
            print("\n[全部完成] 所有模擬、視覺化圖表與技術報告已生成至 output/ 目錄！")
        elif choice == "0":
            print("系統已退出。")
            break
        else:
            print("[無效輸入] 請輸入選單中的有效代碼。")

if __name__ == "__main__":
    main()
