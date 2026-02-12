from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd
import yfinance as yf

from train_model.market_regime import (
    MKT_REGIME_FEATURES,
    compute_market_regime_features,
    merge_regime_into_panel,
)
from train_model.train_model第五版 import compute_rsi


@dataclass(frozen=True)
class FeatureConfig:
    ma_windows: Sequence[int] = (5, 20, 60, 120, 240)
    rsi_windows: Sequence[int] = (120, 240, 420)
    ema_spans: Sequence[int] = (120, 240, 420, 200)


class DataService:
    """
    Data fetch + feature engineering used by the prediction pipeline.

    Column conventions (output):
    - date, open, high, low, close, volume
    """

    def __init__(self, feature_config: FeatureConfig | None = None):
        self._cfg = feature_config or FeatureConfig()

    def fetch_stock_data(
        self,
        symbol: str,
        period: str = "10y",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        if start is not None and end is not None:
            df = yf.download(symbol, start=start, end=end)
        else:
            df = yf.download(symbol, period=period)
        if df.empty or len(df) < 2:
            raise ValueError(
                f"無法取得該股票 ({symbol}) 的歷史資料，請確認代碼正確（台股請用 .TW，如 2330.TW）或稍後再試。"
            )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        ).reset_index()

        # yfinance returns a "Date" column after reset_index()
        df["date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Keep a clean set of base columns, but preserve extra columns for debugging if needed.
        return df

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Moving averages
        for w in self._cfg.ma_windows:
            df[f"ma{w}"] = df["close"].rolling(w).mean()

        # Returns
        df["return_1"] = df["close"].pct_change(1)
        df["return_5"] = df["close"].pct_change(5)

        # RSI
        for w in self._cfg.rsi_windows:
            df[f"rsi_{w}"] = compute_rsi(df["close"], w)

        # EMA
        for s in self._cfg.ema_spans:
            df[f"ema{s}"] = df["close"].ewm(span=s).mean()

        return df

    def _fetch_market_regime_df(
        self,
        date_min: pd.Timestamp,
        date_max: pd.Timestamp,
        market_symbol: str = "^TWII",
        lookback_days: int = 100,
    ) -> pd.DataFrame:
        """下載大盤並計算 regime 特徵，供合併至個股 df。"""
        start = (date_min - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end = date_max.strftime("%Y-%m-%d")
        mkt = yf.download(market_symbol, start=start, end=end, progress=False)
        if mkt is None or mkt.empty:
            raise ValueError(
                f"無法取得大盤 ({market_symbol}) 資料，請檢查網路或代碼。"
            )
        if isinstance(mkt.columns, pd.MultiIndex):
            mkt.columns = mkt.columns.get_level_values(0)
        mkt = mkt.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }).reset_index()
        mkt["date"] = pd.to_datetime(mkt["Date"])
        return compute_market_regime_features(mkt)

    def add_market_regime(
        self,
        df: pd.DataFrame,
        market_symbol: str = "^TWII",
    ) -> pd.DataFrame:
        """
        依 df 的日期範圍取得大盤 regime 特徵並合併。
        推論時單一日期也適用（會抓足夠 lookback 以計算 60 日指標）。
        """
        df = df.copy()
        if "date" not in df.columns:
            raise ValueError("df 需含 date 欄位")
        df["date"] = pd.to_datetime(df["date"])
        date_min, date_max = df["date"].min(), df["date"].max()
        regime_df = self._fetch_market_regime_df(date_min, date_max, market_symbol)
        return merge_regime_into_panel(df, regime_df)

    def required_features(self) -> list[str]:
        ma_cols = [f"ma{w}" for w in self._cfg.ma_windows]
        rsi_cols = [f"rsi_{w}" for w in self._cfg.rsi_windows]
        ema_cols = [f"ema{s}" for s in self._cfg.ema_spans]

        return [
            "open",
            "high",
            "low",
            "close",
            "volume",
            *ma_cols,
            "return_1",
            "return_5",
            *rsi_cols,
            *ema_cols,
            "fund_score",
            *MKT_REGIME_FEATURES,
        ]

