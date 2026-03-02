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


def run_walk_forward_portfolio(
    symbols: list[str],
    start: str,
    end: str,
    run_portfolio_backtest_fn: Callable[[list[str], str | None, str | None], Any],
    *,
    train_years: int = 5,
    test_years: int = 1,
) -> list[WalkForwardResult]:
    """
    執行多資產組合的 Walk-Forward 驗證。

    Parameters
    ----------
    symbols : list[str]
        多個標的代碼（如 ["2330.TW", "0050.TW"]）。
    start : str
        整體區間起日 (YYYY-MM-DD)。
    end : str
        整體區間迄日 (YYYY-MM-DD)。
    run_portfolio_backtest_fn : callable
        組合回測函數，接受 (symbols, start, end) -> BacktestResult。
        例如：backtest_service.run_portfolio_backtest
    train_years : int
        每期訓練年數（目前僅用於劃分視窗）。
    test_years : int
        每期測試年數。

    Returns
    -------
    list[WalkForwardResult]
        各期測試結果（symbol 設為 "PORTFOLIO"）。

    Examples
    --------
    >>> # 測試多資產組合（2330 + 0050）
    >>> wf_results = run_walk_forward_portfolio(
    ...     symbols=["2330.TW", "0050.TW"],
    ...     start="2010-01-01",
    ...     end="2025-12-31",
    ...     run_portfolio_backtest_fn=backtest_service.run_portfolio_backtest,
    ... )
    >>> sharpes = [r.sharpe_ratio for r in wf_results if r.sharpe_ratio is not None]
    >>> print(f"平均 Sharpe: {sum(sharpes) / len(sharpes):.2f}")
    """
    windows = generate_walk_forward_windows(start, end, train_years, test_years)
    results: list[WalkForwardResult] = []
    
    for w in windows:
        try:
            r = run_portfolio_backtest_fn(symbols, w.test_start, w.test_end)
        except Exception:
            continue
        
        results.append(
            WalkForwardResult(
                symbol="PORTFOLIO_" + "_".join(symbols),
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


def compare_walk_forward_strategies(
    symbol: str,
    start: str,
    end: str,
    backtest_service: Any,
    *,
    train_years: int = 5,
    test_years: int = 1,
    portfolio_symbols: list[str] | None = None,
) -> dict:
    """
    比較多種策略的 Walk-Forward 表現。

    測試四種策略變體：
    1. 基準策略（無動量濾鏡）
    2. 動量濾鏡策略（return_5 > 0）
    3. 多資產策略（加入其他標的，如 0050.TW）
    4. 動量濾鏡 + 多資產策略

    Parameters
    ----------
    symbol : str
        主要標的代碼（如 "2330.TW"）。
    start : str
        整體區間起日。
    end : str
        整體區間迄日。
    backtest_service : BacktestService
        回測服務實例（需有 run_backtest 和 run_portfolio_backtest 方法）。
    train_years : int
        每期訓練年數。
    test_years : int
        每期測試年數。
    portfolio_symbols : list[str] | None
        多資產組合的其他標的（如 ["0050.TW"]）。
        若為 None，則只測試單資產策略（1、2）。

    Returns
    -------
    dict
        {
            "baseline": {
                "results": list[WalkForwardResult],
                "sharpe_mean": float,
                "sharpe_std": float,
                "negative_sharpe_pct": float,
                "avg_return": float,
                "avg_mdd": float,
            },
            "momentum": {...},         # 動量濾鏡策略
            "portfolio": {...},        # 多資產策略（若 portfolio_symbols 非 None）
            "momentum_portfolio": {...},  # 動量 + 多資產（若 portfolio_symbols 非 None）
            "summary": pd.DataFrame,   # 各策略比較表
        }

    Examples
    --------
    >>> comparison = compare_walk_forward_strategies(
    ...     symbol="2330.TW",
    ...     start="2010-01-01",
    ...     end="2025-12-31",
    ...     backtest_service=backtest_service,
    ...     portfolio_symbols=["0050.TW"],
    ... )
    >>> print(comparison["summary"])
       strategy              sharpe_mean  sharpe_std  negative_sharpe_pct  avg_mdd
    0  baseline                    1.23        0.85                 0.35   -0.12
    1  momentum                    1.45        0.72                 0.28   -0.10
    2  portfolio                   1.38        0.68                 0.25   -0.09
    3  momentum_portfolio          1.52        0.65                 0.22   -0.08
    """
    import numpy as np
    
    def analyze_results(results: list[WalkForwardResult]) -> dict:
        """分析單一策略的 Walk-Forward 結果。"""
        sharpes = [r.sharpe_ratio for r in results if r.sharpe_ratio is not None]
        returns = [r.annualized_return for r in results]
        mdds = [r.max_drawdown for r in results]
        vols = [r.volatility for r in results]
        
        if not sharpes:
            return {
                "results": results,
                "n_windows": len(results),
                "sharpe_mean": 0.0,
                "sharpe_std": 0.0,
                "negative_sharpe_pct": 1.0,
                "avg_return": 0.0,
                "avg_mdd": 0.0,
                "avg_volatility": 0.0,
            }
        
        negative_count = sum(1 for s in sharpes if s < 0)
        
        return {
            "results": results,
            "n_windows": len(results),
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "negative_sharpe_pct": float(negative_count / len(sharpes)),
            "avg_return": float(np.mean(returns)),
            "avg_mdd": float(np.mean(mdds)),
            "avg_volatility": float(np.mean(vols)),
        }
    
    # 策略 1：基準策略（無動量濾鏡）
    baseline_results = run_walk_forward(
        symbol=symbol,
        start=start,
        end=end,
        run_backtest_fn=lambda sym, s, e: backtest_service.run_backtest(
            symbol=sym, start=s, end=e, momentum_filter=False
        ),
        train_years=train_years,
        test_years=test_years,
    )
    baseline_analysis = analyze_results(baseline_results)
    
    # 策略 2：動量濾鏡策略
    momentum_results = run_walk_forward(
        symbol=symbol,
        start=start,
        end=end,
        run_backtest_fn=lambda sym, s, e: backtest_service.run_backtest(
            symbol=sym, start=s, end=e, momentum_filter=True
        ),
        train_years=train_years,
        test_years=test_years,
    )
    momentum_analysis = analyze_results(momentum_results)
    
    output = {
        "baseline": baseline_analysis,
        "momentum": momentum_analysis,
    }
    
    # 策略 3 & 4：多資產策略（若提供）
    if portfolio_symbols:
        all_symbols = [symbol] + portfolio_symbols
        
        # 策略 3：多資產（無動量濾鏡）
        portfolio_results = run_walk_forward_portfolio(
            symbols=all_symbols,
            start=start,
            end=end,
            run_portfolio_backtest_fn=lambda syms, s, e: backtest_service.run_portfolio_backtest(
                symbols=syms, start=s, end=e
            ),
            train_years=train_years,
            test_years=test_years,
        )
        portfolio_analysis = analyze_results(portfolio_results)
        output["portfolio"] = portfolio_analysis
        
        # 策略 4：動量 + 多資產
        # 注意：目前 run_portfolio_backtest 不支援 momentum_filter 參數
        # 這裡先用基準組合作為替代，實際使用時需擴展 API
        output["momentum_portfolio"] = {
            "results": [],
            "n_windows": 0,
            "sharpe_mean": 0.0,
            "sharpe_std": 0.0,
            "negative_sharpe_pct": 0.0,
            "avg_return": 0.0,
            "avg_mdd": 0.0,
            "avg_volatility": 0.0,
            "note": "動量 + 多資產組合需要擴展 API 支援",
        }
    
    # 生成比較表
    summary_data = []
    for strategy_name, analysis in output.items():
        if strategy_name == "momentum_portfolio" and "note" in analysis:
            continue  # 跳過尚未實現的策略
        summary_data.append({
            "strategy": strategy_name,
            "n_windows": analysis["n_windows"],
            "sharpe_mean": round(analysis["sharpe_mean"], 4),
            "sharpe_std": round(analysis["sharpe_std"], 4),
            "negative_sharpe_pct": round(analysis["negative_sharpe_pct"], 4),
            "avg_return": round(analysis["avg_return"], 6),
            "avg_mdd": round(analysis["avg_mdd"], 6),
            "avg_volatility": round(analysis["avg_volatility"], 6),
        })
    
    output["summary"] = pd.DataFrame(summary_data)
    
    return output
