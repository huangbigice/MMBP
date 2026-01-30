"""
多資產組合風控與權重聚合。

- 對齊多標的日報酬
- 逆波動率權重（等風險貢獻風格）
- 組合層級波動率目標與單一資產曝險上限
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioRiskConfig:
    """組合風控參數。"""

    vol_lookback: int = 20
    """估計各資產波動率之滾動視窗。"""

    target_portfolio_vol_annual: float | None = 0.10
    """組合層級目標年化波動率；None 表示不額外縮放。"""

    max_single_weight: float = 0.40
    """單一資產權重上限，避免過度集中。"""

    min_vol_floor: float = 0.005
    """波動率下限，避免權重爆炸。"""


def align_returns(returns_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """
    將多標的日報酬對齊至共同交易日 index，缺值填 0（視為無曝險）。

    Parameters
    ----------
    returns_dict : dict[str, pd.Series]
        symbol -> 日報酬 Series（index 為 date）。

    Returns
    -------
    pd.DataFrame
        index 為日期，columns 為 symbol，對齊後之日報酬。
    """
    if not returns_dict:
        return pd.DataFrame()

    all_index = returns_dict[next(iter(returns_dict))].index
    for s in returns_dict:
        all_index = all_index.union(returns_dict[s].index)
    all_index = all_index.sort_values().unique()

    aligned = pd.DataFrame(index=all_index)
    for symbol, ser in returns_dict.items():
        aligned[symbol] = ser.reindex(all_index).fillna(0.0)
    return aligned


def compute_inverse_vol_weights(
    returns_df: pd.DataFrame,
    config: PortfolioRiskConfig,
) -> pd.DataFrame:
    """
    依逆波動率計算各資產權重（等風險貢獻風格），並套用單一資產權重上限。

    weight_i ∝ 1/vol_i，再正規化使 sum(weight)=1，且 weight_i <= max_single_weight。

    Parameters
    ----------
    returns_df : pd.DataFrame
        對齊後之日報酬，columns 為 symbol。
    config : PortfolioRiskConfig
        風控參數。

    Returns
    -------
    pd.DataFrame
        與 returns_df 同 index/columns，每行為當日各資產權重。
    """
    vol = returns_df.rolling(config.vol_lookback, min_periods=1).std() * np.sqrt(252)
    vol = vol.clip(lower=config.min_vol_floor)

    inv_vol = 1.0 / vol
    inv_vol = inv_vol.fillna(0.0)

    # 正規化為 sum=1
    row_sum = inv_vol.sum(axis=1).replace(0, np.nan)
    w = inv_vol.div(row_sum, axis=0).fillna(0.0)

    # 單一資產權重上限
    w = w.clip(upper=config.max_single_weight)
    row_sum = w.sum(axis=1).replace(0, np.nan)
    w = w.div(row_sum, axis=0).fillna(0.0)
    return w


def aggregate_portfolio_returns(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    config: PortfolioRiskConfig,
) -> pd.Series:
    """
    依權重聚合組合日報酬，可選組合層級波動率目標縮放。

    Parameters
    ----------
    returns_df : pd.DataFrame
        對齊後之日報酬。
    weights_df : pd.DataFrame
        與 returns_df 同 index/columns 之權重。
    config : PortfolioRiskConfig
        若 target_portfolio_vol_annual 不為 None，則對組合報酬做縮放使 ex-post 滾動波動趨近目標。

    Returns
    -------
    pd.Series
        組合日報酬，index 與 returns_df 相同。
    """
    # 權重與報酬對齊（shift 表示前一日決定之權重用於當日報酬）
    w = weights_df.shift(1).fillna(0.0)
    w = w.reindex_like(returns_df).fillna(0.0)
    r = returns_df.reindex_like(w).fillna(0.0)

    portfolio_return = (w * r).sum(axis=1)

    if config.target_portfolio_vol_annual is None:
        return portfolio_return

    # 組合層級波動率目標：滾動估計組合 vol，再縮放
    roll_vol = portfolio_return.rolling(config.vol_lookback, min_periods=1).std() * np.sqrt(252)
    roll_vol = roll_vol.clip(lower=config.min_vol_floor)
    scale = config.target_portfolio_vol_annual / roll_vol
    scale = scale.clip(upper=2.0)  # 避免過度槓桿
    scaled_return = portfolio_return * scale.shift(1).fillna(1.0)
    return scaled_return
