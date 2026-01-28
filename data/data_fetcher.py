from __future__ import annotations

import pandas as pd
import yfinance as yf


class DataFetcher:
    """
    Backward-compatible fetcher used by the original `main.py` script.

    The new API backend uses `services.data_service.DataService` instead, but we
    keep this class to preserve the existing architecture and reuse later.
    """

    def __init__(self, config=None, logger=None, db=None):
        self._config = config
        self._logger = logger
        self._db = db

    def main(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        symbol = stock_code if "." in stock_code else f"{stock_code}.TW"
        if self._logger:
            self._logger.info("Fetching data: %s %s~%s", symbol, start_date, end_date)
        df = yf.download(symbol, start=start_date, end=end_date)
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
        df["date"] = pd.to_datetime(df["Date"])
        return df.sort_values("date").reset_index(drop=True)

