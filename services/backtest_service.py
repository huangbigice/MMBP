"""
回測服務：訊號產生、波動率目標倉位、多資產組合風控。

- 單資產：可選波動率目標倉位 (Volatility Targeting)，取代固定 100% 倉位。
- 多資產：逆波動率權重 + 組合層級波動率目標與單一資產曝險上限。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config.versioning import ModelVersionInfo
from models.model_loader import ModelLoader
from services.data_service import DataService
from services.backtest import add_adx, compute_position_long_short, compute_regime
from services.risk import (
    PortfolioRiskConfig,
    VolTargetConfig,
    align_returns,
    aggregate_portfolio_returns,
    compute_confidence_vol_weights,
    compute_inverse_vol_weights,
    compute_vol_target_weights,
)

# ---------------------------------------------------------------------------
# 常數與設定
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.6
DEFAULT_HOLD_DAYS = 60
BACKTEST_FUND_SCORE = 0.65
RISK_FREE_RATE = 0.02
FEE_RATE = 0.0015
ATR_STOP_MULTIPLIER = 2.0  # 2 ATR 停損（機構常見）


@dataclass(frozen=True)
class BacktestConfig:
    """單資產回測參數（含是否啟用波動率目標）。"""

    use_vol_targeting: bool = True
    vol_target: VolTargetConfig | None = None  # None 時用預設 VolTargetConfig()


@dataclass(frozen=True)
class BacktestResult:
    symbol: str
    start: str
    end: str
    annualized_return: float
    volatility: float
    max_drawdown: float
    trade_count: int
    sharpe_ratio: float | None
    equity_curve: list[dict[str, Any]]
    model_version: str
    strategy_version: str
    model_effective_date: str


# ---------------------------------------------------------------------------
# 回測服務
# ---------------------------------------------------------------------------


class BacktestService:
    """
    使用 DataService + ModelLoader 執行回測。
    - 支援波動率目標倉位（預設開啟），使策略波動貼近目標，利於風控與客戶溝通。
    - 支援多資產組合回測與逆波動率權重、組合層級波動率目標。
    """

    def __init__(
        self,
        data_service: DataService,
        model_loader: ModelLoader,
        model_version_info: ModelVersionInfo,
    ):
        self._data_service = data_service
        self._model_loader = model_loader
        self._version_info = model_version_info

    def run_backtest(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        use_vol_targeting: bool = True,
        target_vol_annual: float | None = 0.10,
        vol_lookback: int = 20,
        max_leverage: float = 1.0,
        null_mode: None | str = None,
        shuffle_seed: int | None = None,
    ) -> BacktestResult:
        """
        單資產回測。可選波動率目標倉位（預設開啟）。

        null_mode: 用於 Null Model 殺戮測試。
          - None: 真實模型
          - "shuffled": 將 proba_buy 打亂（等同標籤打亂的隨機訊號）
          - "hold": 永遠觀望（signal_long=0, signal_short=0）
        shuffle_seed: null_mode="shuffled" 時可指定隨機種子以便重現。
        """
        vol_config = None
        if use_vol_targeting and target_vol_annual is not None:
            vol_config = VolTargetConfig(
                target_vol_annual=target_vol_annual,
                vol_lookback=vol_lookback,
                max_leverage=max_leverage,
            )

        df = self._build_backtest_df(
            symbol=symbol,
            start=start,
            end=end,
            threshold=threshold,
            vol_config=vol_config,
            null_mode=null_mode,
            shuffle_seed=shuffle_seed,
        )
        return self._result_from_df(df, symbol)

    def get_backtest_df(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        use_vol_targeting: bool = True,
        target_vol_annual: float | None = 0.10,
        vol_lookback: int = 20,
        max_leverage: float = 1.0,
        null_mode: None | str = None,
        shuffle_seed: int | None = None,
    ) -> pd.DataFrame:
        """
        回傳單一標的之回測 DataFrame（含 date, regime, strategy_return, weight 等）。
        供驗證層（如 Regime 生存測試）分析使用。
        """
        vol_config = None
        if use_vol_targeting and target_vol_annual is not None:
            vol_config = VolTargetConfig(
                target_vol_annual=target_vol_annual,
                vol_lookback=vol_lookback,
                max_leverage=max_leverage,
            )
        return self._build_backtest_df(
            symbol=symbol,
            start=start,
            end=end,
            threshold=threshold,
            vol_config=vol_config,
            null_mode=null_mode,
            shuffle_seed=shuffle_seed,
        )

    def run_portfolio_backtest(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        *,
        target_vol_annual: float | None = 0.10,
        vol_lookback: int = 20,
        max_leverage: float = 1.0,
        target_portfolio_vol_annual: float | None = 0.10,
        max_single_weight: float = 0.40,
    ) -> BacktestResult:
        """
        多資產組合回測：各資產先做波動率目標倉位，再以逆波動率權重聚合，
        並可選組合層級波動率目標與單一資產權重上限。
        """
        if len(symbols) == 0:
            raise ValueError("至少需一個標的")
        if len(symbols) == 1:
            return self.run_backtest(
                symbol=symbols[0],
                start=start,
                end=end,
                target_vol_annual=target_vol_annual,
                vol_lookback=vol_lookback,
                max_leverage=max_leverage,
            )

        vol_config = VolTargetConfig(
            target_vol_annual=target_vol_annual or 0.10,
            vol_lookback=vol_lookback,
            max_leverage=max_leverage,
        )
        portfolio_config = PortfolioRiskConfig(
            vol_lookback=vol_lookback,
            target_portfolio_vol_annual=target_portfolio_vol_annual,
            max_single_weight=max_single_weight,
        )

        returns_dict: dict[str, pd.Series] = {}
        for sym in symbols:
            try:
                df = self._build_backtest_df(
                    symbol=sym,
                    start=start,
                    end=end,
                    vol_config=vol_config,
                )
            except Exception:
                continue
            # 以 date 為 index 的日報酬，供組合對齊
            sr = df.set_index("date")["strategy_return"]
            returns_dict[sym] = sr
        if not returns_dict:
            raise ValueError("無任一標的可產出有效回測報酬")

        returns_df = align_returns(returns_dict)
        weights_df = compute_inverse_vol_weights(returns_df, portfolio_config)
        portfolio_return = aggregate_portfolio_returns(
            returns_df, weights_df, portfolio_config
        )

        cum = (1 + portfolio_return).cumprod()
        dd = cum / cum.cummax() - 1
        n = len(cum)
        n_years = n / 252
        total_return = cum.iloc[-1]
        ann_return = total_return ** (1 / n_years) - 1 if n_years > 0 else 0.0
        ann_vol = float(portfolio_return.std() * np.sqrt(252)) if n > 1 else 0.0
        max_dd = float(dd.min())
        sharpe = (
            (ann_return - RISK_FREE_RATE) / ann_vol
            if ann_vol > 0
            else None
        )
        start_str = cum.index.min().strftime("%Y-%m-%d")
        end_str = cum.index.max().strftime("%Y-%m-%d")
        equity_curve = [
            {"date": d.strftime("%Y-%m-%d"), "cumulative_return": float(cum.loc[d])}
            for d in cum.index
        ]

        return BacktestResult(
            symbol="PORTFOLIO",
            start=start_str,
            end=end_str,
            annualized_return=round(ann_return, 6),
            volatility=round(ann_vol, 6),
            max_drawdown=round(max_dd, 6),
            trade_count=0,  # 組合不統計單一交易次數
            sharpe_ratio=round(sharpe, 4) if sharpe is not None else None,
            equity_curve=equity_curve,
            model_version=self._version_info.model_version,
            strategy_version=self._version_info.strategy_version,
            model_effective_date=self._version_info.model_effective_date,
        )

    def _build_backtest_df(
        self,
        symbol: str,
        start: str | None,
        end: str | None,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        vol_config: VolTargetConfig | None = None,
        null_mode: None | str = None,
        shuffle_seed: int | None = None,
    ) -> pd.DataFrame:
        """
        產生單一標的之回測 DataFrame（含 date, strategy_return, cum_return, drawdown 等）。
        供 run_backtest 與 run_portfolio_backtest 使用。

        null_mode: "shuffled" 打亂 proba_buy；"hold" 強制永遠觀望。
        """
        period = "10y" if (start and end) else "5y"
        df = self._data_service.fetch_stock_data(
            symbol=symbol, period=period, start=start, end=end
        )
        if df is None or df.empty:
            raise ValueError("區間內無資料")

        df = self._data_service.add_indicators(df)
        df = self._data_service.add_market_regime(df)
        df["ma_200"] = df["close"].rolling(200).mean()
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(14).mean()
        df["fund_score"] = BACKTEST_FUND_SCORE

        # 多因子 regime 用：ADX、ma_200_slope、regime
        df = add_adx(df, period=14)
        df["ma_200_slope"] = df["ma_200"].diff(5)
        df["regime"] = compute_regime(df, adx_threshold=20, slope_window=5)

        features = self._data_service.required_features()
        df = df.dropna(subset=features).reset_index(drop=True)
        if df.empty:
            raise ValueError("特徵缺失後無可用資料")

        model = self._model_loader.load()
        X = df[features]
        pred = model.predict(X)
        proba = model.predict_proba(X)
        classes = list(getattr(model, "classes_", []))
        idx_buy = classes.index(1) if 1 in classes else 0
        df = df.copy()
        df["proba_buy"] = proba[:, idx_buy]
        df["signal_long"] = ((pred == 1) & (df["fund_score"] > threshold)).astype(int)
        df["signal_short"] = (df["proba_buy"] < 0.5).astype(int)

        # Null Model 殺戮測試：打亂訊號或永遠觀望
        if null_mode == "shuffled":
            rng = np.random.default_rng(shuffle_seed)
            shuffled_proba = df["proba_buy"].values.copy()
            rng.shuffle(shuffled_proba)
            df["proba_buy"] = shuffled_proba
            df["signal_long"] = ((df["proba_buy"] > threshold) & (df["fund_score"] > threshold)).astype(int)
            df["signal_short"] = (df["proba_buy"] < 0.5).astype(int)
        elif null_mode == "hold":
            df["signal_long"] = 0
            df["signal_short"] = 0
            df["proba_buy"] = 0.5

        df["position"] = compute_position_long_short(
            df, atr_mult=ATR_STOP_MULTIPLIER
        )

        # 倉位權重：信心加權波動率目標或單純 position
        if vol_config is not None:
            weight = compute_confidence_vol_weights(
                df["position"], df["proba_buy"], df["return_1"], vol_config
            )
            df["weight"] = weight
            turnover = df["weight"].diff().abs().fillna(0.0)
        else:
            df["weight"] = df["position"].astype(float)
            turnover = df["position"].diff().abs().fillna(0.0)

        df["strategy_return"] = df["weight"].shift(1) * df["return_1"]
        df["strategy_return"] = df["strategy_return"].fillna(0.0)
        df["strategy_return"] -= turnover * FEE_RATE

        df["cum_return"] = (1 + df["strategy_return"]).cumprod()
        df["drawdown"] = df["cum_return"] / df["cum_return"].cummax() - 1
        return df

    def _result_from_df(self, df: pd.DataFrame, symbol: str) -> BacktestResult:
        """從回測 DataFrame 組出 BacktestResult。"""
        total_return = df["cum_return"].iloc[-1]
        n_years = len(df) / 252
        ann_return = total_return ** (1 / n_years) - 1
        ann_vol = float(df["strategy_return"].std() * np.sqrt(252))
        max_dd = float(df["drawdown"].min())
        trade_count = int(((df["position"] != 0) & (df["position"].shift(1) == 0)).sum())
        sharpe = (
            (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else None
        )
        start_str = df["date"].min().strftime("%Y-%m-%d")
        end_str = df["date"].max().strftime("%Y-%m-%d")
        equity_curve = [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "cumulative_return": float(row["cum_return"]),
            }
            for _, row in df[["date", "cum_return"]].iterrows()
        ]
        return BacktestResult(
            symbol=symbol,
            start=start_str,
            end=end_str,
            annualized_return=round(ann_return, 6),
            volatility=round(ann_vol, 6),
            max_drawdown=round(max_dd, 6),
            trade_count=trade_count,
            sharpe_ratio=round(sharpe, 4) if sharpe is not None else None,
            equity_curve=equity_curve,
            model_version=self._version_info.model_version,
            strategy_version=self._version_info.strategy_version,
            model_effective_date=self._version_info.model_effective_date,
        )
