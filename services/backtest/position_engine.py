"""
多空對稱 ATR 進出場引擎。

- 多單：regime == bull 且 signal_long == 1 時進場，2 ATR 移動停損在下方。
- 空單：regime == bear 且 signal_short == 1 時進場，2 ATR 移動停損在上方。
- 輸出 position：1 = 多, -1 = 空, 0 = 空倉。
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _parse_row_signals(row: Any) -> tuple[int, int, int]:
    sig_l = 1 if (pd.notna(row["signal_long"]) and row["signal_long"] == 1) else 0
    sig_s = 1 if (pd.notna(row["signal_short"]) and row["signal_short"] == 1) else 0
    regime = int(row["regime"]) if pd.notna(row["regime"]) else 0
    return sig_l, sig_s, regime


def _update_long_state(
    price: float, atr: float, stop_long: float, atr_mult: float
) -> tuple[bool, float]:
    if price < stop_long:
        return False, 0.0
    new_stop = max(stop_long, price - atr_mult * atr)
    return True, new_stop


def _update_short_state(
    price: float, atr: float, stop_short: float, atr_mult: float
) -> tuple[bool, float]:
    if price > stop_short:
        return False, 0.0
    new_stop = min(stop_short, price + atr_mult * atr)
    return True, new_stop


def _try_enter(
    regime: int, sig_l: int, sig_s: int, price: float, atr: float, atr_mult: float
) -> tuple[int, bool, bool, float, float]:
    """傳回 (position_value, in_long, in_short, stop_long, stop_short)。"""
    if regime == 1 and sig_l == 1:
        return 1, True, False, price - atr_mult * atr, 0.0
    if regime == -1 and sig_s == 1:
        return -1, False, True, 0.0, price + atr_mult * atr
    return 0, False, False, 0.0, 0.0


def compute_position_long_short(
    df: pd.DataFrame,
    *,
    atr_mult: float = 2.0,
) -> pd.Series:
    """
    依 signal_long、signal_short、regime 與 ATR 停損，計算每日倉位 -1 / 0 / 1。

    df 需含欄位：
    - close, ATR
    - signal_long (0/1), signal_short (0/1)
    - regime (1 bull / -1 bear / 0 neutral)

    邏輯：
    - 持多時：若 close < stop_long 則平倉；否則更新 stop_long = max(stop_long, close - atr_mult * ATR)。
    - 持空時：若 close > stop_short 則平倉；否則更新 stop_short = min(stop_short, close + atr_mult * ATR)。
    - 空倉時：regime==1 且 signal_long==1 可做多；regime==-1 且 signal_short==1 可做空。

    Parameters
    ----------
    df : pd.DataFrame
        含上述欄位之回測用 DataFrame。
    atr_mult : float
        ATR 停損倍數，預設 2.0。

    Returns
    -------
    pd.Series
        position：1 / -1 / 0，與 df 同 index。
    """
    required = {"close", "ATR", "signal_long", "signal_short", "regime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df 缺少欄位: {missing}")

    position = pd.Series(0, index=df.index, dtype=int)
    in_long = False
    in_short = False
    stop_long = 0.0
    stop_short = 0.0

    for i in range(len(df)):
        row = df.iloc[i]
        price = row["close"]
        atr = row["ATR"]
        sig_l, sig_s, regime = _parse_row_signals(row)

        if in_long:
            in_long, stop_long = _update_long_state(price, atr, stop_long, atr_mult)
            if in_long:
                position.iloc[i] = 1
        elif in_short:
            in_short, stop_short = _update_short_state(price, atr, stop_short, atr_mult)
            if in_short:
                position.iloc[i] = -1
        else:
            pos_val, new_long, new_short, new_stop_long, new_stop_short = _try_enter(
                regime, sig_l, sig_s, price, atr, atr_mult
            )
            if pos_val != 0:
                position.iloc[i] = pos_val
                in_long, in_short = new_long, new_short
                stop_long, stop_short = new_stop_long, new_stop_short

    return position
