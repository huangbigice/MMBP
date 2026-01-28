from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

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


class PredictionService:
    """
    High-level service that performs:
    - download market data (via DataService)
    - compute indicators
    - fetch fundamental score
    - model inference
    - system rating aggregation
    """

    def __init__(self, model_loader: ModelLoader, data_service: DataService):
        self._model_loader = model_loader
        self._data_service = data_service

    def predict_latest(self, symbol: str, period: str = "10y") -> PredictionResult:
        model = self._model_loader.load()

        df = self._data_service.fetch_stock_data(symbol=symbol, period=period)
        df = self._data_service.add_indicators(df)

        fund_score_dict = get_latest_fund_score(symbol)
        df["fund_score"] = fund_score_dict["total_score"]

        features = self._data_service.required_features()
        df_latest = df.iloc[-1:].copy()

        if df_latest[features].isna().any().any():
            raise ValueError("最新一天特徵有缺失值，資料不足以進行預測")

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
        )

