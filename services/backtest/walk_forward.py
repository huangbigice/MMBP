"""
Walk-forward 回測框架。

Rolling train/test：例如 2010–2014 訓練 → 2015 測試，
避免 in-sample engineering。每期對 test 區間跑回測，彙總各期指標。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

# 依賴由呼叫端注入，避免循環依賴
# BacktestService.run_backtest(symbol, start=..., end=...) -> BacktestResult


@dataclass
class WalkForwardWindow:
    """單一 walk-forward 視窗：訓練區間與測試區間。"""

    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass
class WalkForwardResult:
    """單一視窗的回測結果（與 BacktestResult 對齊）。"""

    symbol: str
    test_start: str
    test_end: str
    annualized_return: float
    volatility: float
    max_drawdown: float
    trade_count: int
    sharpe_ratio: float | None
    equity_curve: list[dict[str, Any]]


def generate_walk_forward_windows(
    start: str,
    end: str,
    train_years: int = 5,
    test_years: int = 1,
) -> list[WalkForwardWindow]:
    """
    產生 rolling 視窗：train_years 訓練、test_years 測試，每年滾動。

    Parameters
    ----------
    start : str
        整體區間起日 (YYYY-MM-DD)。
    end : str
        整體區間迄日 (YYYY-MM-DD)。
    train_years : int
        每期訓練年數。
    test_years : int
        每期測試年數。

    Returns
    -------
    list[WalkForwardWindow]
        視窗列表。
    """
    start_d = pd.Timestamp(start)
    end_d = pd.Timestamp(end)
    windows: list[WalkForwardWindow] = []
    # 以年為單位滾動
    train_end = start_d + pd.DateOffset(years=train_years)
    test_end = train_end + pd.DateOffset(years=test_years)
    while test_end <= end_d:
        windows.append(
            WalkForwardWindow(
                train_start=start_d.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                test_start=train_end.strftime("%Y-%m-%d"),
                test_end=test_end.strftime("%Y-%m-%d"),
            )
        )
        start_d = start_d + pd.DateOffset(years=1)
        train_end = start_d + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
    return windows


def run_walk_forward(
    symbol: str,
    start: str,
    end: str,
    run_backtest_fn: Callable[[str, str | None, str | None], Any],
    *,
    train_years: int = 5,
    test_years: int = 1,
) -> list[WalkForwardResult]:
    """
    執行 walk-forward 回測：每期對 test 區間呼叫 run_backtest_fn，蒐集結果。

    run_backtest_fn(symbol, start, end) 應回傳具 BacktestResult 介面的物件
    （含 annualized_return, volatility, max_drawdown, trade_count, sharpe_ratio, equity_curve 等）。

    Parameters
    ----------
    symbol : str
        標的代碼。
    start : str
        整體區間起日。
    end : str
        整體區間迄日。
    run_backtest_fn : callable
        (symbol, start, end) -> result with BacktestResult-like attributes.
    train_years : int
        每期訓練年數（目前僅用於劃分視窗；實際訓練需由外部串接）。
    test_years : int
        每期測試年數。

    Returns
    -------
    list[WalkForwardResult]
        各期測試結果。
    """
    windows = generate_walk_forward_windows(start, end, train_years, test_years)
    results: list[WalkForwardResult] = []
    for w in windows:
        try:
            r = run_backtest_fn(symbol, w.test_start, w.test_end)
        except Exception:
            continue
        results.append(
            WalkForwardResult(
                symbol=symbol,
                test_start=w.test_start,
                test_end=w.test_end,
                annualized_return=getattr(r, "annualized_return", 0.0),
                volatility=getattr(r, "volatility", 0.0),
                max_drawdown=getattr(r, "max_drawdown", 0.0),
                trade_count=getattr(r, "trade_count", 0),
                sharpe_ratio=getattr(r, "sharpe_ratio", None),
                equity_curve=getattr(r, "equity_curve", []),
            )
        )
    return results
