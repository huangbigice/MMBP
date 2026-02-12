"""
大盤市場狀態（regime）特徵：供訓練與推論使用。

以指數（如 ^TWII）計算趨勢、波動、回撤等，讓模型具備 regime awareness。
"""

from __future__ import annotations

import pandas as pd


# 推論與訓練共用的特徵名稱，需與 DataService.required_features() 一致
MKT_REGIME_FEATURES = [
    "mkt_ma20_slope",
    "mkt_ma60_slope",
    "mkt_vol_60d",
    "mkt_drawdown_60",
    "mkt_close_over_ma20",
    "mkt_close_over_ma60",
]


def compute_market_regime_features(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    依大盤日線計算市場狀態特徵。

    假設 market_df 含欄位：date（或 index 為 DatetimeIndex）、close。
    若為 yfinance 下載後 rename 的格式，需有 close、且 date 為欄位或 index。

    Parameters
    ----------
    market_df : pd.DataFrame
        大盤日線，需含 close；若有 Date 則會用於 date，否則用 index。

    Returns
    -------
    pd.DataFrame
        每交易日一列，含 date 及 MKT_REGIME_FEATURES 所列欄位。
    """
    df = market_df.copy()
    if hasattr(df.index, "normalize"):
        df = df.reset_index()
    if "Date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Date": "date"})
    if "date" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "date"})
    if "date" not in df.columns:
        raise ValueError("market_df 需有 date 欄位或 DatetimeIndex")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    if "close" not in df.columns and "Close" in df.columns:
        df["close"] = df["Close"]
    close = df["close"]
    # 均線
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    # 斜率：5 日變動（簡化），與訓練一致
    df["mkt_ma20_slope"] = ma20.pct_change(5)
    df["mkt_ma60_slope"] = ma60.pct_change(5)
    # 60 日波動率（日報酬年化前之日標準差）
    df["mkt_vol_60d"] = close.pct_change().rolling(60).std()
    # 自 60 日高點以來的回撤
    roll_max = close.rolling(60, min_periods=1).max()
    df["mkt_drawdown_60"] = (close - roll_max) / roll_max.replace(0, pd.NA)
    # 收盤相對均線位置
    df["mkt_close_over_ma20"] = (close / ma20) - 1
    df["mkt_close_over_ma60"] = (close / ma60) - 1

    out = df[["date"] + MKT_REGIME_FEATURES].copy()
    out = out.dropna(subset=MKT_REGIME_FEATURES)
    return out


def merge_regime_into_panel(
    panel_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    on: str = "date",
    *,
    ffill_regime: bool = True,
) -> pd.DataFrame:
    """
    將大盤 regime 特徵以日期合併進 panel。

    若個股最新日期在大盤尚未有當日資料時，left join 會使該列 regime 為 NaN。
    ffill_regime=True（預設）時會依日期向前填補，讓「最新一天」沿用最近一筆有效 regime。

    Parameters
    ----------
    panel_df : pd.DataFrame
        含 on 欄位（通常為 date）的訓練/回測 panel。
    regime_df : pd.DataFrame
        compute_market_regime_features 的輸出（含 date 與 MKT_REGIME_FEATURES）。
    on : str
        合併鍵，預設 "date"。
    ffill_regime : bool
        是否對 regime 欄位做向前填補，避免最新列缺值。

    Returns
    -------
    pd.DataFrame
        合併後的 panel；ffill_regime 時 regime 缺值會用前一筆有效值填滿。
    """
    regime_df = regime_df.copy()
    regime_df[on] = pd.to_datetime(regime_df[on]).dt.normalize()
    panel_df = panel_df.copy()
    panel_df[on] = pd.to_datetime(panel_df[on]).dt.normalize()
    merged = panel_df.merge(
        regime_df[[on] + MKT_REGIME_FEATURES],
        on=on,
        how="left",
    )
    if ffill_regime and merged[MKT_REGIME_FEATURES].isna().any().any():
        merged = merged.sort_values(on)
        merged[MKT_REGIME_FEATURES] = (
            merged[MKT_REGIME_FEATURES].ffill().bfill()
        )
    return merged
