"""
波動率目標倉位 (Volatility Targeting)。

依 ex-ante 波動率動態調整倉位，使策略曝險波動貼近目標，
符合機構風控與客戶可接受風險水準。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolTargetConfig:
    """波動率目標參數。"""

    target_vol_annual: float = 0.10
    """目標年化波動率，例如 0.10 = 10%。"""

    vol_lookback: int = 20
    """估計波動率之滾動視窗（交易日）。"""

    max_leverage: float = 1.0
    """單一資產最大槓桿（倉位上限）。"""

    min_vol_floor: float = 0.01
    """波動率下限，避免除零與極端放大。"""


def compute_vol_target_weights(
    position_binary: pd.Series,
    returns: pd.Series,
    config: VolTargetConfig,
) -> pd.Series:
    """
    依波動率目標計算每日倉位權重。

    公式：weight_t = min(target_vol / vol_estimate_t, max_leverage) when position_t==1 else 0。
    vol_estimate 為 returns 之滾動年化標準差（僅用於有倉位時之縮放）。

    Parameters
    ----------
    position_binary : pd.Series
        0/1 倉位訊號，index 需與 returns 對齊。
    returns : pd.Series
        標的日報酬（例如 return_1）。
    config : VolTargetConfig
        目標波動率與上下限參數。

    Returns
    -------
    pd.Series
        每日權重，與輸入同 index。
    """
    vol = returns.rolling(config.vol_lookback, min_periods=1).std()
    # 年化：日波動 * sqrt(252)
    vol_annual = vol * np.sqrt(252)
    vol_annual = vol_annual.clip(lower=config.min_vol_floor)

    # 目標權重 = target_vol / vol_estimate，再以 max_leverage 與 0 夾住
    raw_weight = config.target_vol_annual / vol_annual
    weight = raw_weight.clip(upper=config.max_leverage)
    weight = weight * position_binary
    return weight.fillna(0.0)
