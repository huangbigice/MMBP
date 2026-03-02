"""
尾端風險與壓力測試。

- Monte Carlo：對歷史報酬 bootstrap 重抽，產生多條 equity curve，取回撤分布。
- Block bootstrap：保留短期相關性。
- Stress scenario：固定極端區間或單日衝擊（如 -15%），重跑策略反應。
- 歷史危機時期測試：模擬 2022 熊市、2020 COVID 崩盤等特定時期表現。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ==============================================================================
# 歷史危機場景定義
# ==============================================================================

CRISIS_SCENARIOS = {
    "2022_bear": {
        "name": "2022年熊市",
        "start": "2022-01-01",
        "end": "2022-12-31",
        "description": "升息週期導致科技股大幅修正",
    },
    "2020_covid": {
        "name": "2020年COVID崩盤",
        "start": "2020-02-20",
        "end": "2020-03-23",
        "description": "疫情恐慌性拋售",
    },
    "2018_correction": {
        "name": "2018年修正",
        "start": "2018-10-01",
        "end": "2018-12-31",
        "description": "貿易戰與Fed升息雙重壓力",
    },
    "2008_crisis": {
        "name": "2008年金融海嘯",
        "start": "2008-09-01",
        "end": "2009-03-31",
        "description": "雷曼倒閉引發全球金融危機",
    },
}


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


def stress_test_period(
    df: pd.DataFrame,
    scenario_key: str | None = None,
    custom_start: str | None = None,
    custom_end: str | None = None,
    risk_free_rate: float = 0.02,
) -> dict:
    """
    對特定歷史時期執行壓力測試，提取該時期的表現指標。

    Parameters
    ----------
    df : pd.DataFrame
        完整回測結果，需含欄位：date, strategy_return, cum_return, drawdown。
        建議使用 BacktestService.get_backtest_df() 產生。
    scenario_key : str | None
        預設場景鍵值（如 "2022_bear"），查詢 CRISIS_SCENARIOS。
        若為 None，則使用 custom_start/end。
    custom_start : str | None
        自訂時期起始日（YYYY-MM-DD），當 scenario_key=None 時使用。
    custom_end : str | None
        自訂時期結束日（YYYY-MM-DD），當 scenario_key=None 時使用。
    risk_free_rate : float
        無風險利率，用於計算 Sharpe Ratio。

    Returns
    -------
    dict
        {
            "scenario": str,
            "name": str,
            "start": str,
            "end": str,
            "total_return": float,
            "max_drawdown": float,
            "volatility": float,
            "sharpe": float | None,
            "description": str,
            "n_days": int,
        }

    Raises
    ------
    ValueError
        場景不存在或時期範圍無資料。

    Examples
    --------
    >>> # 測試 2022 熊市表現
    >>> result = stress_test_period(df, scenario_key="2022_bear")
    >>> print(f"MDD: {result['max_drawdown']:.2%}")
    MDD: -8.5%
    """
    # 決定時期範圍
    if scenario_key is not None:
        if scenario_key not in CRISIS_SCENARIOS:
            raise ValueError(
                f"未知場景: {scenario_key}。"
                f"可用場景: {list(CRISIS_SCENARIOS.keys())}"
            )
        scenario = CRISIS_SCENARIOS[scenario_key]
        start_date = scenario["start"]
        end_date = scenario["end"]
        scenario_name = scenario["name"]
        description = scenario["description"]
    else:
        if custom_start is None or custom_end is None:
            raise ValueError("當 scenario_key=None 時，必須提供 custom_start 和 custom_end")
        start_date = custom_start
        end_date = custom_end
        scenario_name = f"自訂時期 {start_date} 至 {end_date}"
        description = "使用者自訂壓力測試時期"
        scenario_key = "custom"

    # 確保 date 為 datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

    # 篩選時期
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    period_df = df[mask].copy()

    if period_df.empty:
        raise ValueError(
            f"時期 {start_date} 至 {end_date} 無資料。"
            f"回測範圍: {df['date'].min()} 至 {df['date'].max()}"
        )

    # 計算指標
    n_days = len(period_df)
    returns = period_df["strategy_return"]
    
    # 總報酬（從該時期起點累積）
    cum_ret = (1 + returns).cumprod()
    total_return = float(cum_ret.iloc[-1] - 1) if not cum_ret.empty else 0.0

    # 最大回撤（該時期內）
    peak = cum_ret.cummax()
    dd = cum_ret / peak - 1
    max_drawdown = float(dd.min()) if not dd.empty else 0.0

    # 波動率（年化）
    volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0

    # Sharpe Ratio（年化）
    if volatility > 0:
        n_years = n_days / 252
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        sharpe = (ann_return - risk_free_rate) / volatility
    else:
        sharpe = None

    return {
        "scenario": scenario_key,
        "name": scenario_name,
        "start": start_date,
        "end": end_date,
        "total_return": round(total_return, 6),
        "max_drawdown": round(max_drawdown, 6),
        "volatility": round(volatility, 6),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "description": description,
        "n_days": n_days,
    }


def stress_test_multiple_periods(
    df: pd.DataFrame,
    scenarios: list[str] | None = None,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """
    對多個危機時期執行壓力測試，返回比較表。

    Parameters
    ----------
    df : pd.DataFrame
        完整回測結果（需含 date, strategy_return, cum_return, drawdown）。
    scenarios : list[str] | None
        場景列表（如 ["2022_bear", "2020_covid"]）。
        若為 None，測試所有預設場景。
    risk_free_rate : float
        無風險利率。

    Returns
    -------
    pd.DataFrame
        各場景的指標比較表，欄位包含：
        scenario, name, start, end, total_return, max_drawdown,
        volatility, sharpe, description, n_days。

    Examples
    --------
    >>> results = stress_test_multiple_periods(df)
    >>> print(results[["name", "max_drawdown", "sharpe"]])
       name            max_drawdown  sharpe
    0  2022年熊市         -0.085     1.2
    1  2020年COVID崩盤    -0.123     0.8
    """
    if scenarios is None:
        scenarios = list(CRISIS_SCENARIOS.keys())

    results = []
    for scenario_key in scenarios:
        try:
            result = stress_test_period(
                df,
                scenario_key=scenario_key,
                risk_free_rate=risk_free_rate,
            )
            results.append(result)
        except ValueError as e:
            # 若該時期無資料，記錄但繼續
            print(f"警告：場景 {scenario_key} 測試失敗 - {e}")
            continue

    if not results:
        raise ValueError("無任何場景可產出測試結果")

    return pd.DataFrame(results)


def analyze_stress_distribution(
    results: list[MonteCarloResult],
    mdd_threshold: float = -0.10,
) -> dict:
    """
    分析 Monte Carlo 壓力測試結果分佈，計算風險指標。

    Parameters
    ----------
    results : list[MonteCarloResult]
        monte_carlo_bootstrap() 或 block_bootstrap() 的輸出。
    mdd_threshold : float
        MDD 警示門檻（預設 -10%）。

    Returns
    -------
    dict
        {
            "n_simulations": int,                    # 模擬次數
            "mdd_mean": float,                       # MDD 平均值
            "mdd_std": float,                        # MDD 標準差
            "mdd_percentiles": {5: ..., 50: ..., 95: ...},  # MDD 分位數
            "prob_mdd_below_threshold": float,       # MDD < threshold 的機率
            "return_mean": float,                    # 總報酬平均值
            "return_std": float,                     # 總報酬標準差
            "return_percentiles": {5: ..., 50: ..., 95: ...},
            "var_95": float,                         # 95% VaR（報酬）
            "cvar_95": float,                        # 95% CVaR（報酬）
            "worst_case": MonteCarloResult,          # 最差路徑
            "best_case": MonteCarloResult,           # 最佳路徑
        }

    Examples
    --------
    >>> mc_results = monte_carlo_bootstrap(returns, n_sim=5000)
    >>> analysis = analyze_stress_distribution(mc_results, mdd_threshold=-0.10)
    >>> print(f"MDD < -10% 機率: {analysis['prob_mdd_below_threshold']:.1%}")
    MDD < -10% 機率: 12.3%
    """
    if not results:
        raise ValueError("結果列表為空")

    n_sim = len(results)
    mdds = np.array([r.max_drawdown for r in results])
    returns = np.array([r.total_return for r in results])

    # MDD 統計
    mdd_mean = float(np.mean(mdds))
    mdd_std = float(np.std(mdds))
    mdd_pct = {
        5: float(np.percentile(mdds, 5)),
        50: float(np.percentile(mdds, 50)),
        95: float(np.percentile(mdds, 95)),
    }
    prob_below_threshold = float(np.sum(mdds < mdd_threshold) / n_sim)

    # 報酬統計
    return_mean = float(np.mean(returns))
    return_std = float(np.std(returns))
    return_pct = {
        5: float(np.percentile(returns, 5)),
        50: float(np.percentile(returns, 50)),
        95: float(np.percentile(returns, 95)),
    }

    # VaR 與 CVaR（95% 信賴水準，左尾風險）
    var_95 = return_pct[5]
    cvar_95 = float(np.mean(returns[returns <= var_95]))

    # 極端情況
    worst_idx = int(np.argmin(mdds))
    best_idx = int(np.argmax(returns))
    worst_case = results[worst_idx]
    best_case = results[best_idx]

    return {
        "n_simulations": n_sim,
        "mdd_mean": round(mdd_mean, 6),
        "mdd_std": round(mdd_std, 6),
        "mdd_percentiles": {k: round(v, 6) for k, v in mdd_pct.items()},
        "prob_mdd_below_threshold": round(prob_below_threshold, 4),
        "return_mean": round(return_mean, 6),
        "return_std": round(return_std, 6),
        "return_percentiles": {k: round(v, 6) for k, v in return_pct.items()},
        "var_95": round(var_95, 6),
        "cvar_95": round(cvar_95, 6),
        "worst_case": worst_case,
        "best_case": best_case,
    }
