"""
尾端風險與壓力測試。

- Monte Carlo：對歷史報酬 bootstrap 重抽，產生多條 equity curve，取回撤分布。
- Block bootstrap：保留短期相關性。
- Stress scenario：固定極端區間或單日衝擊（如 -15%），重跑策略反應。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    """單一模擬路徑的摘要。"""

    total_return: float
    max_drawdown: float
    final_equity: float


def monte_carlo_bootstrap(
    returns: pd.Series,
    n_sim: int = 1000,
    seed: int | None = None,
) -> list[MonteCarloResult]:
    """
    對日報酬做 bootstrap 重抽，產生 n_sim 條 equity curve，回傳每條的總報酬與最大回撤。

    Parameters
    ----------
    returns : pd.Series
        歷史日報酬（例如 strategy_return 或 return_1）。
    n_sim : int
        模擬次數。
    seed : int, optional
        隨機種子。

    Returns
    -------
    list[MonteCarloResult]
        每條路徑的 total_return、max_drawdown、final_equity。
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n == 0:
        return []
    results: list[MonteCarloResult] = []
    for _ in range(n_sim):
        idx = rng.integers(0, n, size=n)
        sim_ret = returns.iloc[idx].values
        cum = np.cumprod(1 + sim_ret)
        total_return = cum[-1] - 1
        peak = np.maximum.accumulate(cum)
        dd = cum / peak - 1
        max_dd = float(np.min(dd))
        results.append(
            MonteCarloResult(
                total_return=float(total_return),
                max_drawdown=max_dd,
                final_equity=float(cum[-1]),
            )
        )
    return results


def block_bootstrap(
    returns: pd.Series,
    block_size: int,
    n_sim: int = 1000,
    seed: int | None = None,
) -> list[MonteCarloResult]:
    """
    Block bootstrap：以連續 block_size 日為一塊重抽，保留短期相關性。

    Parameters
    ----------
    returns : pd.Series
        歷史日報酬。
    block_size : int
        區塊長度（交易日）。
    n_sim : int
        模擬次數。
    seed : int, optional
        隨機種子。

    Returns
    -------
    list[MonteCarloResult]
        每條路徑的摘要。
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n < block_size:
        return monte_carlo_bootstrap(returns, n_sim=n_sim, seed=seed)
    n_blocks = (n + block_size - 1) // block_size
    results: list[MonteCarloResult] = []
    for _ in range(n_sim):
        blocks = []
        for _ in range(n_blocks):
            start = int(rng.integers(0, n - block_size + 1))
            blocks.append(returns.iloc[start : start + block_size].values)
        sim_ret = np.concatenate(blocks)[:n]
        cum = np.cumprod(1 + sim_ret)
        total_return = cum[-1] - 1
        peak = np.maximum.accumulate(cum)
        dd = cum / peak - 1
        max_dd = float(np.min(dd))
        results.append(
            MonteCarloResult(
                total_return=float(total_return),
                max_drawdown=max_dd,
                final_equity=float(cum[-1]),
            )
        )
    return results


def stress_single_day_shock(
    returns: pd.Series,
    shock_date: str | pd.Timestamp,
    shock_return: float,
) -> pd.Series:
    """
    在指定日期植入單日報酬衝擊（例如 -0.15），其餘不變。

    用於檢視策略在極端單日（跳空、limit down）下的權益與回撤反應。

    Parameters
    ----------
    returns : pd.Series
        歷史日報酬，index 為日期。
    shock_date : str or pd.Timestamp
        要植入衝擊的日期。
    shock_return : float
        該日取代後的報酬（例如 -0.15 表示 -15%）。

    Returns
    -------
    pd.Series
        植入衝擊後的報酬序列，與 returns 同 index。
    """
    out = returns.copy()
    if isinstance(shock_date, str):
        shock_date = pd.Timestamp(shock_date)
    if shock_date in out.index:
        out.loc[shock_date] = shock_return
    return out
