from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd
import yfinance as yf

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
        ]

