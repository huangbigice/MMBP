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

from config import StrategyConfig
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

# 與 strategy_config.yaml backtest 預設一致，供驗證腳本等處匯入使用
FEE_RATE = 0.0015
RISK_FREE_RATE = 0.02


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
    # 品質評級資訊（可選）
    quality_rating: str | None = None
    quality_label: str | None = None
    portfolio_eligible: bool | None = None


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
        strategy_config: StrategyConfig,
    ):
        self._data_service = data_service
        self._model_loader = model_loader
        self._version_info = model_version_info
        self._config = strategy_config

    def run_backtest(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        *,
        threshold: float | None = None,
        use_vol_targeting: bool = True,
        target_vol_annual: float | None = None,
        vol_lookback: int | None = None,
        max_leverage: float | None = None,
        momentum_filter: bool = False,
        null_mode: None | str = None,
        shuffle_seed: int | None = None,
    ) -> BacktestResult:
        """
        單資產回測。可選波動率目標倉位（預設開啟）。
        
        **重要**：要完全對齊訓練模型邏輯，請設置 `use_vol_targeting=False`：
        ```python
        result = backtest_service.run_backtest(
            symbol="2330.TW",
            use_vol_targeting=False  # 關閉波動率目標，完全對齊訓練模型
        )
        ```
        
        訓練模型使用的確切邏輯：
        - position = proba_strong.shift(1).clip(0, 0.5)  # 連續概率控倉
        - weight = position  # 不使用波動率目標
        - strategy_return = weight.shift(1) * return_1
        - 單日跌>5%時當日報酬=0

        Parameters
        ----------
        threshold : float | None
            模型機率門檻。None 時使用配置預設值。
        use_vol_targeting : bool
            是否啟用波動率目標倉位。預設為 True。
            **注意**：訓練模型不使用波動率目標，要完全對齊請設置為 False。
        target_vol_annual : float | None
            目標年化波動率。None 時使用配置預設值（18%）。
        vol_lookback : int | None
            波動率估計回望期。None 時使用配置預設值。
        max_leverage : float | None
            最大槓桿。None 時使用配置預設值。
        momentum_filter : bool
            是否啟用動量濾鏡（return_5 > 0）。預設為 False。
        null_mode : None | str
            用於 Null Model 殺戮測試。
            - None: 真實模型
            - "shuffled": 將 proba_buy 打亂（等同標籤打亂的隨機訊號）
            - "hold": 永遠觀望（signal_long=0, signal_short=0）
        shuffle_seed : int | None
            null_mode="shuffled" 時可指定隨機種子以便重現。
        """
        # 從配置讀取預設值
        if threshold is None:
            threshold = self._config.strategy.model_threshold
        if target_vol_annual is None:
            target_vol_annual = self._config.volatility_targeting.default_target_annual
        if vol_lookback is None:
            vol_lookback = self._config.volatility_targeting.vol_lookback
        if max_leverage is None:
            max_leverage = self._config.volatility_targeting.max_leverage
        
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
            momentum_filter=momentum_filter,
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
        threshold: float | None = None,
        use_vol_targeting: bool = True,
        target_vol_annual: float | None = None,
        vol_lookback: int | None = None,
        max_leverage: float | None = None,
        momentum_filter: bool = False,
        null_mode: None | str = None,
        shuffle_seed: int | None = None,
    ) -> pd.DataFrame:
        """
        回傳單一標的之回測 DataFrame（含 date, regime, strategy_return, weight 等）。
        供驗證層（如 Regime 生存測試）分析使用。
        """
        # 從配置讀取預設值
        if threshold is None:
            threshold = self._config.strategy.model_threshold
        if target_vol_annual is None:
            target_vol_annual = self._config.volatility_targeting.default_target_annual
        if vol_lookback is None:
            vol_lookback = self._config.volatility_targeting.vol_lookback
        if max_leverage is None:
            max_leverage = self._config.volatility_targeting.max_leverage
        
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
            momentum_filter=momentum_filter,
            null_mode=null_mode,
            shuffle_seed=shuffle_seed,
        )

    def run_portfolio_backtest(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        *,
        target_vol_annual: float | None = None,
        vol_lookback: int | None = None,
        max_leverage: float | None = None,
        target_portfolio_vol_annual: float | None = None,
        max_single_weight: float | None = None,
    ) -> BacktestResult:
        """
        多資產組合回測：各資產先做波動率目標倉位，再以逆波動率權重聚合，
        並可選組合層級波動率目標與單一資產權重上限。
        """
        # 從配置讀取預設值
        if target_vol_annual is None:
            target_vol_annual = self._config.volatility_targeting.default_target_annual
        if vol_lookback is None:
            vol_lookback = self._config.volatility_targeting.vol_lookback
        if max_leverage is None:
            max_leverage = self._config.volatility_targeting.max_leverage
        if target_portfolio_vol_annual is None:
            target_portfolio_vol_annual = self._config.portfolio.target_portfolio_vol_annual
        if max_single_weight is None:
            max_single_weight = self._config.portfolio.max_single_weight
        
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
            target_vol_annual=target_vol_annual,
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
            (ann_return - self._config.backtest.risk_free_rate) / ann_vol
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
        threshold: float | None = None,
        vol_config: VolTargetConfig | None = None,
        momentum_filter: bool = False,  # 已移除：對齊訓練模型邏輯後不再使用動量過濾
        null_mode: None | str = None,
        shuffle_seed: int | None = None,
    ) -> pd.DataFrame:
        """
        產生單一標的之回測 DataFrame（含 date, strategy_return, cum_return, drawdown 等）。
        供 run_backtest 與 run_portfolio_backtest 使用。
        
        **已對齊訓練模型邏輯（train_model第六版.py）**：
        - 使用 proba_strong（類別1概率）作為連續信號，而非硬性0/1信號
        - 移除 fund_score 依賴（訓練模型已移除此特徵）
        - 簡化信號生成：移除 regime/ATR/動量過濾
        - 單日停損：return_1 < -0.05 時當日報酬=0

        null_mode: "shuffled" 打亂 proba_buy；"hold" 強制永遠觀望。
        """
        # 訓練模型使用的最大部位和停損閾值（對齊 train_model第六版.py）
        MAX_POSITION = 0.5
        LOSS_THRESHOLD = -0.05  # 單日跌超過 5% 時停損
        # 訓練模型使用的概率門檻（用於統計 signal_long，不影響實際部位）
        # **注意**：對齊訓練模型邏輯，使用 0.55（訓練模型的 PROBA_THRESHOLD）
        # threshold 參數保留用於 API 兼容性，但實際使用固定的 0.55（對齊訓練模型）
        TRAIN_PROBA_THRESHOLD = 0.55
        
        period = "10y" if (start and end) else "5y"
        df = self._data_service.fetch_stock_data(
            symbol=symbol, period=period, start=start, end=end
        )
        if df is None or df.empty:
            raise ValueError("區間內無資料")

        df = self._data_service.add_indicators(df)
        df = self._data_service.add_market_regime(df)
        
        # **移除：不再需要 ATR、regime、fund_score（訓練模型已移除）**

        # 從模型文件中獲取特徵列表和 scaler（如果存在）
        model = self._model_loader.load()
        model_features = self._model_loader.get_features()
        scaler = self._model_loader.get_scaler()
        
        # **關鍵修正：確保特徵順序與訓練模型完全一致**
        # 如果模型保存了特徵列表，必須使用該順序（與訓練時一致）
        if model_features is not None:
            features = model_features  # 直接使用模型保存的特徵順序
        else:
            # 向後相容：舊模型可能沒有保存特徵列表
            features = self._data_service.required_features(model_features=None)
        
        # **終極診斷：MKT_REGIME_FEATURES 合併驗證**
        from train_model.market_regime import MKT_REGIME_FEATURES
        mkt_features_present = [f for f in MKT_REGIME_FEATURES if f in df.columns]
        mkt_features_missing = [f for f in MKT_REGIME_FEATURES if f not in df.columns]
        
        if mkt_features_missing:
            import warnings
            warnings.warn(
                f"🚨 MKT_REGIME_FEATURES 缺失！缺少：{mkt_features_missing}\n"
                f"  這會導致模型預測完全失效（proba_strong 全<0.4）\n"
                f"  請檢查 add_market_regime() 是否正確執行，或日期格式是否一致。"
            )
        
        # 檢查 MKT_REGIME_FEATURES 的 NaN 比例
        if mkt_features_present:
            mkt_nan_ratio = df[mkt_features_present].isna().mean().mean()
            if mkt_nan_ratio > 0.1:  # 如果超過10%是NaN
                import warnings
                warnings.warn(
                    f"⚠️ MKT_REGIME_FEATURES NaN比例過高：{mkt_nan_ratio:.1%}\n"
                    f"  這可能導致模型預測異常。建議檢查日期合併邏輯。"
                )
        
        # 檢查所有特徵是否存在
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(
                f"缺少必要特徵：{missing_features[:10]}{'...' if len(missing_features) > 10 else ''}。\n"
                f"MKT_REGIME_FEATURES狀態：存在={len(mkt_features_present)}/{len(MKT_REGIME_FEATURES)}，"
                f"缺失={mkt_features_missing}\n"
                f"請確認 data_service.add_indicators() 和 add_market_regime() 已正確執行。"
            )
        
        df = df.dropna(subset=features).reset_index(drop=True)
        if df.empty:
            raise ValueError("特徵缺失後無可用資料")
        
        # **關鍵修正：對齊訓練模型邏輯，並同時支援分類 / 迴歸模型**
        # 必須按照 features 順序提取（與訓練時 FEATURES 一致）
        X = df[features].fillna(0)

        # **特徵一致性驗證（用於調試）**
        if model_features is not None:
            # 驗證特徵名稱匹配
            if set(X.columns) != set(model_features):
                missing = set(model_features) - set(X.columns)
                extra = set(X.columns) - set(model_features)
                raise ValueError(
                    f"特徵名稱不匹配！缺少：{missing}，多餘：{extra}。"
                    f"請確認 data_service.add_indicators() 和 add_market_regime() 已正確執行。"
                )
            # 確保順序一致（scaler / 模型都依賴特徵順序）
            X = X[model_features]

        # ----- 依模型型別建立 proba_strong / proba_buy -----
        df = df.copy()

        if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
            # 分類模型（含 predict_proba）
            # 若 scaler 存在，視為特徵標準化器（如 StandardScaler）
            if scaler is not None and hasattr(scaler, "transform"):
                X_input = scaler.transform(X)
            else:
                X_input = X.values

            proba = model.predict_proba(X_input)
            classes = list(getattr(model, "classes_", []))

            # 使用類別 1（長期持有）的機率作為 proba_strong
            idx_strong = classes.index(1) if 1 in classes else 0
            df["proba_strong"] = proba[:, idx_strong]
            df["proba_buy"] = df["proba_strong"]

        else:
            # 迴歸模型：例如 RandomForestRegressor，輸出連續目標
            # 假設 scaler（若存在）是將目標映射到 [0,1] 的 MinMaxScaler
            import numpy as np

            y_pred = model.predict(X.values).astype(float)
            if scaler is not None and hasattr(scaler, "transform"):
                proba_cont = scaler.transform(y_pred.reshape(-1, 1)).ravel()
            else:
                proba_cont = y_pred

            # 將值限制在 [0,1] 範圍內
            proba_cont = np.clip(proba_cont, 0.0, 1.0)
            df["proba_strong"] = proba_cont
            df["proba_buy"] = proba_cont
        
        # **終極診斷：proba_strong 分佈驗證**
        # 正常情況下，proba_strong 應該有合理的分佈：
        # - mean ≈ 0.18-0.25（因為標籤1約佔18%）
        # - max ≈ 0.70-0.90
        # - 超過0.55的比例 ≈ 5-10%
        proba_stats = {
            "mean": float(df["proba_strong"].mean()),
            "std": float(df["proba_strong"].std()),
            "min": float(df["proba_strong"].min()),
            "max": float(df["proba_strong"].max()),
            "above_055": float((df["proba_strong"] > 0.55).mean()),
            "above_060": float((df["proba_strong"] > 0.60).mean()),
            "above_040": float((df["proba_strong"] > 0.40).mean()),
        }
        
        # **終極診斷：根據 proba_strong 分佈判斷問題**
        import warnings
        if proba_stats["max"] < 0.45:
            # 致命錯誤：proba全<0.45，特徵工程徹底失敗
            warnings.warn(
                f"🚨 致命錯誤：proba_strong 全<0.45（max={proba_stats['max']:.3f}）\n"
                f"  特徵工程徹底失敗！最可能原因：\n"
                f"  1. MKT_REGIME_FEATURES 缺失或NaN過多（95%概率）\n"
                f"  2. scaler 標準化失敗（特徵順序錯亂）\n"
                f"  3. 特徵計算方式與訓練不一致\n"
                f"  診斷資訊：\n"
                f"  - 特徵數量：{len(features)}\n"
                f"  - MKT_REGIME存在：{len(mkt_features_present)}/{len(MKT_REGIME_FEATURES)}\n"
                f"  - proba_strong mean={proba_stats['mean']:.3f}, max={proba_stats['max']:.3f}"
            )
        elif proba_stats["max"] < 0.55:
            # 警告：proba全<0.55，交易信號過少
            warnings.warn(
                f"⚠️ proba_strong 異常低（max={proba_stats['max']:.3f}），交易信號可能過少\n"
                f"  可能原因：\n"
                f"  1. MKT_REGIME_FEATURES NaN過多\n"
                f"  2. 特徵工程與訓練模型不完全一致\n"
                f"  3. scaler 範圍外（單股特徵分佈≠多股票面板）\n"
                f"  診斷：>0.55比例={proba_stats['above_055']:.1%}, >0.4比例={proba_stats['above_040']:.1%}"
            )
        elif proba_stats["above_055"] < 0.01:
            # 警告：交易信號過少（<1%）
            warnings.warn(
                f"⚠️ 交易信號過少：>0.55比例={proba_stats['above_055']:.1%}\n"
                f"  雖然 proba_strong max={proba_stats['max']:.3f} 正常，但超過0.55的比例過低\n"
                f"  可能原因：MKT_REGIME_FEATURES 部分缺失或NaN"
            )
        
        # Null Model 殺戮測試：打亂訊號或永遠觀望
        if null_mode == "shuffled":
            rng = np.random.default_rng(shuffle_seed)
            shuffled_proba = df["proba_strong"].values.copy()
            rng.shuffle(shuffled_proba)
            df["proba_strong"] = shuffled_proba
            df["proba_buy"] = shuffled_proba
        elif null_mode == "hold":
            df["proba_strong"] = 0.0
            df["proba_buy"] = 0.0

        # **核心修正：100%對齊訓練模型邏輯**
        # 訓練模型確切公式：position = proba_strong.shift(1).clip(0, MAX_POSITION)
        # 這是連續概率控倉，不是硬性0/1信號！
        df["position"] = df["proba_strong"].shift(1).clip(0, MAX_POSITION)
        df["position"] = df["position"].fillna(0.0)
        
        # **關鍵診斷輸出：驗證信號邏輯是否正確**
        import warnings
        position_stats = {
            "mean": float(df["position"].mean()),
            "max": float(df["position"].max()),
            "min": float(df["position"].min()),
            "above_zero": float((df["position"] > 0).mean()),
            "above_01": float((df["position"] > 0.1).mean()),
            "above_02": float((df["position"] > 0.2).mean()),
        }
        
        # 如果 position > 0 的比例太低，說明 proba_strong 異常低
        if position_stats["above_zero"] < 0.5:
            warnings.warn(
                f"🚨 警告：position > 0 比例過低（{position_stats['above_zero']:.1%}）\n"
                f"  這表示 proba_strong 異常低，可能原因：\n"
                f"  1. MKT_REGIME_FEATURES 缺失或NaN過多\n"
                f"  2. 特徵工程與訓練不一致\n"
                f"  3. scaler 標準化失敗\n"
                f"  診斷：proba_strong mean={proba_stats['mean']:.3f}, "
                f"position mean={position_stats['mean']:.3f}"
            )
        elif position_stats["mean"] < 0.1:
            warnings.warn(
                f"⚠️ 警告：position 平均值過低（{position_stats['mean']:.3f}）\n"
                f"  訓練模型預期平均 position ≈ 0.27（27%）\n"
                f"  當前平均值過低，可能導致交易次數偏少"
            )
        
        # 向後相容：保留 signal_long 欄位（用於統計），但實際部位由 position 控制
        # **注意**：signal_long 僅用於統計交易次數，實際部位由 position（連續概率）控制
        # 使用訓練模型的門檻 0.55 進行統計（對齊訓練邏輯）
        df["signal_long"] = (df["proba_strong"] > TRAIN_PROBA_THRESHOLD).astype(int)
        df["signal_short"] = 0  # 訓練模型只做多，不做空

        # 倉位權重：波動率目標或單純 position
        # **關鍵**：訓練模型不使用波動率目標，直接使用 position
        # 如果 use_vol_targeting=False，weight = position（對齊訓練模型）
        if vol_config is not None:
            # 波動率目標會調整 weight，可能導致交易次數減少
            weight = compute_confidence_vol_weights(
                df["position"], df["proba_strong"], df["return_1"], vol_config
            )
            df["weight"] = weight
            turnover = df["weight"].diff().abs().fillna(0.0)
        else:
            # **對齊訓練模型：直接使用 position 作為 weight**
            df["weight"] = df["position"].astype(float)
            turnover = df["position"].diff().abs().fillna(0.0)

        # **對齊訓練模型：簡單停損邏輯**
        # 訓練模型：strategy_return = position * return_1，但 return_1 < -0.05 時當日報酬=0
        df["strategy_return"] = df["weight"].shift(1) * df["return_1"]
        df["strategy_return"] = df["strategy_return"].fillna(0.0)
        # 單日跌>5%時當日報酬清零（對齊訓練模型）
        df.loc[df["return_1"] < LOSS_THRESHOLD, "strategy_return"] = 0
        # 扣除手續費
        df["strategy_return"] -= turnover * self._config.backtest.fee_rate

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
        
        # **對齊訓練模型：交易次數定義**
        # 訓練模型使用 buy_signal = proba_strong > 0.55 的次數作為交易次數
        # 而不是 position 從0變為非0的次數（因為連續概率控倉，position 幾乎總是非0）
        if "signal_long" in df.columns:
            # 使用 signal_long（proba_strong > 0.55）的次數
            trade_count = int(df["signal_long"].sum())
        else:
            # 向後相容：如果沒有 signal_long，使用 position 變化次數
            trade_count = int(((df["position"] != 0) & (df["position"].shift(1) == 0)).sum())
        sharpe = (
            (ann_return - self._config.backtest.risk_free_rate) / ann_vol if ann_vol > 0 else None
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
