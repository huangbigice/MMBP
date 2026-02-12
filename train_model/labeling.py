"""
標籤建構：絕對門檻與相對排名（橫斷面）。

相對標籤讓模型學「當天哪幾檔相對最強」，產出的 proba_buy 具橫斷面排序意義。
"""

from __future__ import annotations

import pandas as pd


def build_absolute_labels(
    df: pd.DataFrame,
    return_col: str,
    *,
    high_threshold: float = 0.10,
    low_threshold: float = -0.05,
    middle_low: float = 0.05,
) -> pd.Series:
    """
    絕對報酬門檻標籤（供對照或相容用）。

    規則：return > high_threshold → 1，return < low_threshold → 0，其餘 → 2。

    Parameters
    ----------
    df : pd.DataFrame
        需含 return_col。
    return_col : str
        未來報酬欄位名。
    high_threshold : float
        超過此值標為 1（長期持有）。
    low_threshold : float
        低於此值標為 0（不建議）。
    middle_low : float
        介於 middle_low 與 high_threshold 之間為 2（觀望）。

    Returns
    -------
    pd.Series
        與 df 同 index，值為 0 / 1 / 2。
    """
    s = df[return_col]
    label = pd.Series(2, index=df.index, dtype=int)
    label.loc[s > high_threshold] = 1
    label.loc[s < low_threshold] = 0
    return label


def build_relative_labels(
    df: pd.DataFrame,
    horizon_col: str,
    *,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
    min_stocks_per_date: int = 5,
) -> pd.DataFrame:
    """
    依當日橫斷面排名建構相對標籤。

    同一交易日內，依 horizon_col 對所有股票排名：
    - 排名前 top_pct（例如前 20%）→ 1（買入/長期持有）
    - 排名後 bottom_pct（例如後 20%）→ 0（不建議）
    - 中間 → 2（觀望）

    當日樣本數少於 min_stocks_per_date 的日期會整日剔除，避免極端日失真。

    Parameters
    ----------
    df : pd.DataFrame
        合併後 panel，需含 date 與 horizon_col。
    horizon_col : str
        未來報酬欄位名（如 future_return_30）。
    top_pct : float
        前多少比例標為 1。
    bottom_pct : float
        後多少比例標為 0。
    min_stocks_per_date : int
        當日至少幾檔才納入排名，否則該日所有列剔除。

    Returns
    -------
    pd.DataFrame
        僅保留有排名的列，並新增欄位 "label"（0/1/2）、"rank_pct"（當日百分位，0~1）。
    """
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("df 需含 date 欄位")
    if horizon_col not in df.columns:
        raise ValueError(f"df 需含 {horizon_col} 欄位")

    # 當日樣本數
    date_counts = df.groupby("date").size()
    valid_dates = date_counts[date_counts >= min_stocks_per_date].index
    df = df[df["date"].isin(valid_dates)].copy()

    # 當日橫斷面百分位排名（1 = 最高報酬）
    df["rank_pct"] = df.groupby("date")[horizon_col].rank(pct=True, method="average")

    # 相對標籤：前 top_pct → 1，後 bottom_pct → 0，其餘 → 2
    label = pd.Series(2, index=df.index, dtype=int)
    label[df["rank_pct"] >= (1 - top_pct)] = 1
    label[df["rank_pct"] <= bottom_pct] = 0
    df["label"] = label

    return df
