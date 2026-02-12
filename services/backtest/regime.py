"""
多因子 Regime 分類：bull / bear / neutral。

使用 ma_200、ma_200_slope、ADX 降低假突破與側盤時的誤判，
僅在明確趨勢時開倉。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    計算 ADX（Average Directional Index）並寫入 DataFrame。

    ADX 衡量趨勢強度，與方向無關。需欄位：high, low, close。

    Parameters
    ----------
    df : pd.DataFrame
        含 high, low, close 的價格資料。
    period : int
        Wilder 平滑週期，預設 14。

    Returns
    -------
    pd.DataFrame
        新增欄位 "ADX" 的 copy（不 mutate 原 df）。
    """
    df = df.copy()
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # Wilder smoothing (RMA): first value = first raw, then (prev * (n-1) + curr) / n
    def _rma(series: pd.Series, n: int) -> pd.Series:
        out = series.ewm(alpha=1 / n, adjust=False).mean()
        return out

    atr = _rma(tr, period)
    plus_di = 100 * _rma(plus_dm, period) / atr
    minus_di = 100 * _rma(minus_dm, period) / atr

    # DX and ADX
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = np.where(di_sum > 0, 100 * di_diff / di_sum, 0.0)
    dx = pd.Series(dx, index=df.index)
    adx = _rma(dx, period)
    df["ADX"] = adx
    return df


def compute_regime(
    df: pd.DataFrame,
    *,
    adx_threshold: float = 20.0,
    slope_window: int = 5,
) -> pd.Series:
    """
    多因子 regime：1 = bull, -1 = bear, 0 = neutral。

    條件：
    - Bull: price > ma_200 且 ma_200_slope > 0 且 ADX > adx_threshold
    - Bear: price < ma_200 且 ma_200_slope < 0 且 ADX > adx_threshold
    - Neutral: 其餘（無趨勢或方向不一致）

    Parameters
    ----------
    df : pd.DataFrame
        需含 close, ma_200, ADX。ma_200_slope 若不存在會自動以 ma_200.diff(slope_window) 計算。
    adx_threshold : float
        ADX 門檻，僅高於此值視為有趨勢。
    slope_window : int
        ma_200 斜率計算的差分天數。

    Returns
    -------
    pd.Series
        與 df 同 index，值為 1 / -1 / 0。
    """
    if "ma_200" not in df.columns:
        raise ValueError("df 需含 ma_200")
    if "ADX" not in df.columns:
        raise ValueError("df 需含 ADX，請先呼叫 add_adx(df)")

    close = df["close"]
    ma = df["ma_200"]
    adx = df["ADX"]

    if "ma_200_slope" in df.columns:
        slope = df["ma_200_slope"]
    else:
        slope = ma.diff(slope_window)

    has_trend = adx > adx_threshold
    above_ma = close > ma
    below_ma = close < ma
    ma_rising = slope > 0
    ma_falling = slope < 0

    bull = has_trend & above_ma & ma_rising
    bear = has_trend & below_ma & ma_falling

    regime = pd.Series(0, index=df.index)
    regime = regime.astype(int)
    regime[bull] = 1
    regime[bear] = -1
    return regime
