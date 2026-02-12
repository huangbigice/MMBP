#!/usr/bin/env python3
"""
五層 Alpha 驗證測試 — 執行腳本。

使用方式（在專案根目錄、已安裝依賴的環境下）：
  python run_five_layer_tests.py
  python run_five_layer_tests.py --symbol 2330.TW --start 2018-01-01 --end 2025-12-31

依序執行：
  第一層：Null Model 殺戮測試（真實 vs 打亂 vs 永遠觀望）
  第二層：Walk-forward 穩定性（Sharpe 分佈）
  第三層：Regime 生存測試（多頭/空頭/高波動/低波動）
  第四層：Benchmark 智商測試（B&H、Momentum、MA cross、RSI(14)>50）
  第五層：Economic Plausibility 自評表
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="五層 Alpha 驗證測試")
    parser.add_argument("--symbol", default="2330.TW", help="標的代碼")
    parser.add_argument("--start", default="2018-01-01", help="回測區間起日")
    parser.add_argument("--end", default="2025-12-31", help="回測區間迄日")
    parser.add_argument("--wf-start", default="2010-01-01", help="Walk-forward 整體區間起日")
    parser.add_argument("--no-plot", action="store_true", help="不繪製 Layer 2 Sharpe 分佈圖")
    args = parser.parse_args()

    from validation.five_layer_tests import run_all_five_layers

    run_all_five_layers(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        wf_start=args.wf_start,
        plot_layer2=not args.no_plot,
    )


if __name__ == "__main__":
    main()
