from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from config.versioning import ModelVersionInfo
from models.model_loader import ModelLoader
from services.data_service import DataService
from system_rating import get_latest_fund_score, system_rating


LABEL_MAP: Mapping[int, str] = {
    0: "不建議持有",
    1: "長期持有",
    2: "觀望",
}


@dataclass(frozen=True)
class PredictionResult:
    symbol: str
    probabilities: dict[str, float]
    system_score: float
    tech_score: float
    fund_score: float
    proba_buy: float
    recommendation: str
    model_version: str
    strategy_version: str
    model_effective_date: str


class PredictionService:
    """
    High-level service that performs:
    - download market data (via DataService)
    - compute indicators
    - fetch fundamental score
    - model inference
    - system rating aggregation

    部分股票無法預測常見原因：歷史資料不足（需約 420 交易日）、
    資料源暫時異常或該日有缺漏；同一股票偶爾失敗再試一次可成功，多為資料源不穩定。
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        data_service: DataService,
        model_version_info: ModelVersionInfo,
    ):
        self._model_loader = model_loader
        self._data_service = data_service
        self._version_info = model_version_info

    def predict_latest(self, symbol: str, period: str = "10y") -> PredictionResult:
        model = self._model_loader.load()

        df = self._data_service.fetch_stock_data(symbol=symbol, period=period)
        df = self._data_service.add_indicators(df)
        df = self._data_service.add_market_regime(df)

        fund_score_dict = get_latest_fund_score(symbol)
        raw_fund = fund_score_dict.get("total_score")
        df["fund_score"] = 0.5 if raw_fund is None or pd.isna(raw_fund) else raw_fund

        features = self._data_service.required_features()
        df_latest = df.iloc[-1:].copy()

        # 若最新一筆有缺值，改用「最後一筆特徵完整的列」做預測，避免多數請求因單日缺值而失敗
        if df_latest[features].isna().any().any():
            complete = df.dropna(subset=features)
            if complete.empty:
                missing = df_latest[features].columns[df_latest[features].isna().any()].tolist()
                raise ValueError(
                    "特徵有缺失值，資料不足以進行預測。"
                    f"缺失欄位：{missing[:10]}{'...' if len(missing) > 10 else ''}。"
                    "可能原因：該股票歷史資料不足（需至少約 420 個交易日）、最近有缺漏，或資料源暫時異常。"
                )
            df_latest = complete.iloc[-1:].copy()

        X = df_latest[features]

        proba = model.predict_proba(X)[0]
        classes = list(getattr(model, "classes_", []))
        if not classes:
            raise ValueError("模型缺少 classes_，無法對應輸出類別")

        probabilities: dict[str, float] = {}
        for i, c in enumerate(classes):
            label = LABEL_MAP.get(int(c), str(c))
            probabilities[label] = float(round(float(proba[i]), 6))

        if 1 not in [int(c) for c in classes]:
            raise ValueError("模型輸出未包含類別 1（長期持有），無法計算 proba_buy")
        proba_buy = float(proba[classes.index(1)])

        df_latest["proba_buy"] = proba_buy
        rated_df = system_rating(df_latest)
        row = rated_df.iloc[0]

        return PredictionResult(
            symbol=symbol,
            probabilities=probabilities,
            system_score=float(row["system_score"]),
            tech_score=float(row["tech_score"]),
            fund_score=float(row["fund_score"]),
            proba_buy=float(row["proba_buy"]),
            recommendation=str(row["recommendation"]),
            model_version=self._version_info.model_version,
            strategy_version=self._version_info.strategy_version,
            model_effective_date=self._version_info.model_effective_date,
        )

