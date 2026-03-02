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
from train_model.train_model第六版 import compute_rsi


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
        interval: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV time series.

        - Daily (default): interval=None, use yfinance period/start/end.
        - Intraday: interval in {"1m","5m","10m","30m","60m"}.
          Note: yfinance doesn't support 10m directly; we resample 1m -> 10m.
        """

        is_intraday = interval is not None
        normalized_interval = interval

        if is_intraday:
            allowed = {"1m", "5m", "10m", "30m", "60m"}
            if normalized_interval not in allowed:
                raise ValueError(f"不支援的 interval: {normalized_interval}（允許：{sorted(allowed)}）")

        if start is not None and end is not None:
            if is_intraday and normalized_interval == "10m":
                raw = yf.download(symbol, start=start, end=end, interval="1m")
                df = self._resample_ohlcv(raw, rule="10min")
            else:
                if normalized_interval is None:
                    df = yf.download(symbol, start=start, end=end)
                else:
                    df = yf.download(symbol, start=start, end=end, interval=normalized_interval)
        else:
            if is_intraday and normalized_interval == "10m":
                raw = yf.download(symbol, period=period, interval="1m")
                df = self._resample_ohlcv(raw, rule="10min")
            else:
                if normalized_interval is None:
                    df = yf.download(symbol, period=period)
                else:
                    df = yf.download(symbol, period=period, interval=normalized_interval)
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

        # yfinance returns either "Date" (daily) or "Datetime" (intraday) after reset_index()
        time_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
        if time_col is None:
            # Fallback: use the first column (reset_index output) as timestamp.
            time_col = df.columns[0]
        df["date"] = pd.to_datetime(df[time_col])
        df = df.sort_values("date").reset_index(drop=True)
        
        # **數據質量檢查**
        # 檢查是否有異常值或缺失值
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                # 檢查是否有負值或零值（價格不應該為負或零）
                if (df[col] <= 0).any():
                    import warnings
                    warnings.warn(
                        f"⚠️ {symbol} 的 {col} 欄位包含非正值（負值或零），"
                        f"共 {(df[col] <= 0).sum()} 筆。這可能表示數據異常。"
                    )
                # 檢查是否有異常大的跳動（單日變化超過50%可能是數據錯誤）
                if col == "close":
                    pct_change = df[col].pct_change().abs()
                    extreme_changes = (pct_change > 0.5).sum()
                    if extreme_changes > 0:
                        import warnings
                        warnings.warn(
                            f"⚠️ {symbol} 的 {col} 欄位有 {extreme_changes} 筆單日變化超過50%，"
                            f"這可能表示數據異常或除權除息。"
                        )
        
        # **關鍵修正：對齊訓練模型邏輯（僅日線）**
        # 訓練模型在計算特徵前對原始數據進行 quantile clipping（1%-99%）。
        # 分鐘級 K 線不應套用此裁剪，避免蠟燭線 wick 被削平。
        if not is_intraday:
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    # 只在有足夠數據時才進行 clipping（避免小樣本時過度裁剪）
                    if len(df) > 100:
                        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
                        # 記錄裁剪前的統計，以便調試
                        original_min, original_max = df[col].min(), df[col].max()
                        df[col] = df[col].clip(lo, hi).astype(float)
                        # 如果裁剪幅度過大，發出警告
                        if (original_min < lo * 0.5) or (original_max > hi * 1.5):
                            import warnings
                            warnings.warn(
                                f"⚠️ {symbol} 的 {col} 欄位進行了大幅裁剪："
                                f"原始範圍=[{original_min:.2f}, {original_max:.2f}], "
                                f"裁剪後=[{lo:.2f}, {hi:.2f}]。"
                                f"這可能移除了真實的極端值或除權除息調整。"
                            )
                    else:
                        # 數據量少時，只轉換類型，不進行 clipping
                        df[col] = df[col].astype(float)
        else:
            # Intraday: ensure numeric dtypes only.
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 檢查最終數據質量
        if df["close"].isna().any():
            import warnings
            warnings.warn(
                f"⚠️ {symbol} 的 close 欄位仍有 NaN 值，共 {df['close'].isna().sum()} 筆。"
            )
        
        # Keep a clean set of base columns, but preserve extra columns for debugging if needed.
        return df

    @staticmethod
    def _resample_ohlcv(raw: pd.DataFrame, rule: str) -> pd.DataFrame:
        """
        Resample yfinance OHLCV DataFrame by time rule (e.g. "10min").

        raw: yfinance output, index must be DatetimeIndex.
        """
        if raw is None or raw.empty:
            return raw

        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        needed = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"分鐘資料缺少欄位：{missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        out = (
            df[needed]
            .resample(rule)
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
        )
        out = out.dropna(subset=["Close"])
        return out

    def get_quote(self, symbol: str) -> dict[str, str | float]:
        """
        使用 yfinance 取得即時／延遲報價。
        回傳 current_price, previous_close, change, change_percent；取不到資料時 raise ValueError。
        """
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if not info:
            raise ValueError(
                f"無法取得該股票 ({symbol}) 的報價資料，請確認代碼正確（台股請用 .TW，如 2330.TW）或稍後再試。"
            )
        current = info.get("currentPrice") or info.get("regularMarketPrice")
        if current is None:
            current = info.get("previousClose")
        previous = info.get("previousClose")
        if previous is None:
            previous = current
        if current is None or previous is None:
            raise ValueError(
                f"無法取得該股票 ({symbol}) 的價格欄位，請確認代碼正確或稍後再試。"
            )
        try:
            current_f = float(current)
            previous_f = float(previous)
        except (TypeError, ValueError):
            raise ValueError(
                f"該股票 ({symbol}) 的價格資料格式異常，請稍後再試。"
            )
        if previous_f <= 0:
            change = 0.0
            change_pct = 0.0
        else:
            change = current_f - previous_f
            change_pct = (change / previous_f) * 100.0

        result: dict[str, str | float] = {
            "symbol": symbol,
            "current_price": current_f,
            "previous_close": previous_f,
            "change": change,
            "change_percent": change_pct,
        }
        # 可選：當日 open/high/low，供最後一根 K 線更新用
        open_val = info.get("regularMarketOpen") or info.get("open")
        high_val = info.get("regularMarketDayHigh") or info.get("dayHigh")
        low_val = info.get("regularMarketDayLow") or info.get("dayLow")
        if open_val is not None:
            try:
                result["open"] = float(open_val)
            except (TypeError, ValueError):
                pass
        if high_val is not None:
            try:
                result["high"] = float(high_val)
            except (TypeError, ValueError):
                pass
        if low_val is not None:
            try:
                result["low"] = float(low_val)
            except (TypeError, ValueError):
                pass
        return result

    def apply_quote_to_last_row(self, df: pd.DataFrame, quote: dict) -> pd.DataFrame:
        """
        用即時報價更新 df 最後一筆的 close（及可選的 open/high/low）。
        不回傳指標重算，由呼叫端決定是否再跑 add_indicators。
        """
        if df.empty:
            return df
        out = df.copy()
        last = out.index[-1]
        out.loc[last, "close"] = float(quote["current_price"])
        if "open" in quote:
            out.loc[last, "open"] = float(quote["open"])
        if "high" in quote:
            out.loc[last, "high"] = float(quote["high"])
        if "low" in quote:
            out.loc[last, "low"] = float(quote["low"])
        return out

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Moving averages（需與訓練腳本一致）
        for w in self._cfg.ma_windows:
            df[f"ma{w}"] = df["close"].rolling(w).mean()

        # Returns（含訓練時使用的 return_20）
        df["return_1"] = df["close"].pct_change(1)
        df["return_5"] = df["close"].pct_change(5)
        df["return_20"] = df["close"].pct_change(20)

        # RSI（含訓練時使用的 rsi_14）
        for w in self._cfg.rsi_windows:
            df[f"rsi_{w}"] = compute_rsi(df["close"], w)
        df["rsi_14"] = compute_rsi(df["close"], 14)

        # 訓練時使用的均線比例特徵（ma5_close, ma20_close, ma60_close）
        df["ma5_close"] = df["close"] / df["ma5"]
        df["ma20_close"] = df["close"] / df["ma20"]
        df["ma60_close"] = df["close"] / df["ma60"]

        # 訓練時使用的高頻特徵
        # price_strength: 10 日價格強度（百分位排名）
        df["price_strength"] = (df["close"] / df["close"].shift(10)).rank(pct=True)
        # volatility_regime: 20日波動率（百分位排名）
        df["volatility_regime"] = df["return_1"].rolling(20).std().rank(pct=True)
        # volume_spike: 成交量異常（相對於20日均量，上限3倍）
        df["volume_spike"] = (
            df["volume"] / df["volume"].rolling(20).mean()
        ).clip(0, 3)

        # EMA（雖然新模型未必使用，但保留以維持相容性）
        for s in self._cfg.ema_spans:
            df[f"ema{s}"] = df["close"].ewm(span=s).mean()

        # ------------------------
        # 長期動量與趨勢特徵（新模型需要）
        # ------------------------
        # mom_60 / 120 / 240：長期動量
        df["mom_60"] = df["close"].pct_change(60)
        df["mom_120"] = df["close"].pct_change(120)
        df["mom_240"] = df["close"].pct_change(240)

        # ma*_slope：均線斜率（與訓練腳本一致的窗口）
        df["ma20_slope"] = df["ma20"].diff(5)   # 約一週斜率
        df["ma60_slope"] = df["ma60"].diff(20)  # 約一個月斜率
        df["ma120_slope"] = df["ma120"].diff(60)  # 約三個月斜率

        # ------------------------
        # 穩定性特徵（新模型需要）
        # ------------------------
        # 年化波動 vol_1y
        df["vol_1y"] = df["return_1"].rolling(252).std() * (252 ** 0.5)

        # 最大回撤 max_drawdown_1y（252 日滾動視窗）
        rolling_max = df["close"].rolling(252, min_periods=1).max()
        df["max_drawdown_1y"] = (df["close"] / rolling_max - 1).rolling(252).min()

        # 月上漲比率 monthly_positive_ratio
        if "date" not in df.columns:
            raise ValueError("df 需含 date 欄位以計算 monthly_positive_ratio")
        df_idx = df.set_index("date")
        monthly_return = df_idx["close"].resample("ME").last().pct_change()
        monthly_positive = (monthly_return > 0).astype(int)
        # 將月度結果對齊回日度索引，並做 12 個月滾動平均
        monthly_positive = monthly_positive.reindex(df["date"], method="ffill")
        df["monthly_positive_ratio"] = monthly_positive.rolling(12).mean().values

        # ------------------------
        # 市場比較特徵（beta_1y、relative_strength_1y）
        # 單檔股票版本，邏輯與訓練腳本一致。
        # 若大盤下載失敗，仍會建立欄位但為 NaN，避免「not in index」錯誤。
        # ------------------------
        try:
            date_min, date_max = df["date"].min(), df["date"].max()
            # 多抓一點 lookback 以確保 252 日滾動視窗可計算
            mkt_start = (date_min - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
            mkt_end = date_max.strftime("%Y-%m-%d")
            mkt = yf.download("^TWII", start=mkt_start, end=mkt_end, progress=False)
            if mkt is not None and not mkt.empty:
                if isinstance(mkt.columns, pd.MultiIndex):
                    mkt.columns = mkt.columns.get_level_values(0)
                mkt = mkt.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                ).reset_index()
                mkt["date"] = pd.to_datetime(mkt["Date"])
                mkt = mkt[["date", "close"]].sort_values("date").reset_index(drop=True)
                mkt["mkt_return_1"] = mkt["close"].pct_change()

                # 依日期對齊個股與大盤
                merged = (
                    df[["date", "return_1", "close"]]
                    .merge(
                        mkt.rename(columns={"close": "mkt_close"}),
                        on="date",
                        how="left",
                    )
                    .sort_values("date")
                )

                # beta_1y：252 日滾動協方差 / 變異數
                cov = merged["return_1"].rolling(252).cov(merged["mkt_return_1"])
                var = merged["mkt_return_1"].rolling(252).var()
                beta = cov / var

                # relative_strength_1y：個股 1 年報酬 - 大盤 1 年報酬
                stock_ret_1y = merged["close"].pct_change(252)
                mkt_ret_1y = merged["mkt_close"].pct_change(252)
                rel = stock_ret_1y - mkt_ret_1y

                df["beta_1y"] = beta.values
                df["relative_strength_1y"] = rel.values
            else:
                import warnings

                warnings.warn(
                    "⚠️ 無法下載大盤指數資料以計算 beta_1y / relative_strength_1y，"
                    "相關特徵將為 NaN。"
                )
                df["beta_1y"] = pd.NA
                df["relative_strength_1y"] = pd.NA
        except Exception as e:
            import warnings

            warnings.warn(
                f"⚠️ 計算 beta_1y / relative_strength_1y 失敗：{e}；"
                "相關特徵將為 NaN。"
            )
            df["beta_1y"] = pd.NA
            df["relative_strength_1y"] = pd.NA

        return df

    def _fetch_market_regime_df(
        self,
        date_min: pd.Timestamp,
        date_max: pd.Timestamp,
        market_symbol: str = "^TWII",
        lookback_days: int = 100,
    ) -> pd.DataFrame | None:
        """
        下載大盤並計算 regime 特徵，供合併至個股 df。
        
        Returns
        -------
        pd.DataFrame | None
            大盤 regime 特徵 DataFrame，如果下載失敗則返回 None。
        """
        try:
            start = (date_min - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            end = date_max.strftime("%Y-%m-%d")
            mkt = yf.download(market_symbol, start=start, end=end, progress=False)
            if mkt is None or mkt.empty:
                return None
            
            if isinstance(mkt.columns, pd.MultiIndex):
                mkt.columns = mkt.columns.get_level_values(0)
            # 若扁平化後出現重複欄位名，只保留每組重複中的第一個，避免後續 df["close"] 變成多欄
            if mkt.columns.duplicated().any():
                mkt = mkt.loc[:, ~mkt.columns.duplicated()]
            mkt = mkt.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }).reset_index()
            mkt["date"] = pd.to_datetime(mkt["Date"])
            return compute_market_regime_features(mkt)
        except Exception as e:
            import warnings
            warnings.warn(
                f"⚠️ 下載大盤 regime 資料失敗（{market_symbol}）：{str(e)}\n"
                f"  MKT_REGIME_FEATURES 將無法合併，這會導致模型預測失效！"
            )
            return None

    def add_market_regime(
        self,
        df: pd.DataFrame,
        market_symbol: str = "^TWII",
    ) -> pd.DataFrame:
        """
        依 df 的日期範圍取得大盤 regime 特徵並合併。
        推論時單一日期也適用（會抓足夠 lookback 以計算 60 日指標）。
        
        **關鍵修正**：確保日期格式一致，避免 merge 失敗導致 MKT_REGIME_FEATURES 缺失。
        """
        df = df.copy()
        if "date" not in df.columns:
            raise ValueError("df 需含 date 欄位")
        
        # **關鍵修正：統一日期格式**
        # 確保日期格式與 merge_regime_into_panel 一致（normalize後比較）
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        date_min, date_max = df["date"].min(), df["date"].max()
        
        # 下載大盤資料（需要額外的 lookback 以計算 60 日指標）
        regime_df = self._fetch_market_regime_df(date_min, date_max, market_symbol)
        
        # **關鍵修正：驗證 regime_df 不為空**
        if regime_df is None or regime_df.empty:
            import warnings
            warnings.warn(
                f"⚠️ 無法取得大盤 regime 資料（{market_symbol}），"
                f"MKT_REGIME_FEATURES 將全部為 NaN！"
            )
            # 創建空的 MKT_REGIME_FEATURES 以避免後續錯誤
            from train_model.market_regime import MKT_REGIME_FEATURES
            for feat in MKT_REGIME_FEATURES:
                df[feat] = pd.NA
            return df
        
        # 合併大盤 regime 特徵
        merged = merge_regime_into_panel(df, regime_df)
        
        # **終極驗證：檢查 MKT_REGIME_FEATURES 是否成功合併**
        from train_model.market_regime import MKT_REGIME_FEATURES
        mkt_features_present = [f for f in MKT_REGIME_FEATURES if f in merged.columns]
        if len(mkt_features_present) < len(MKT_REGIME_FEATURES):
            import warnings
            warnings.warn(
                f"⚠️ MKT_REGIME_FEATURES 合併不完整："
                f"存在={len(mkt_features_present)}/{len(MKT_REGIME_FEATURES)}"
            )
        
        return merged

    def required_features(self, model_features: list[str] | None = None) -> list[str]:
        """
        返回模型預測所需的特徵列表。
        
        Parameters
        ----------
        model_features : list[str] | None
            模型文件中保存的特徵列表。如果提供，則使用此列表；
            否則返回預設特徵列表（向後相容）。
        
        Returns
        -------
        list[str]
            特徵名稱列表。
        """
        # 如果提供了模型特徵列表，直接使用
        if model_features is not None:
            return model_features
        
        # 向後相容：返回預設特徵列表
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

