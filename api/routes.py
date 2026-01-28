from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    HealthResponse,
    IndicatorsResponse,
    PredictionRequest,
    PredictionResponse,
    StockDataResponse,
)
from services.data_service import DataService
from services.prediction_service import PredictionService
 

router = APIRouter(prefix="/api/v1")


def get_prediction_service(request: Request) -> PredictionService:
    svc = getattr(request.app.state, "prediction_service", None)
    if svc is None:
        raise RuntimeError("PredictionService not initialized")
    return svc


def get_data_service(request: Request) -> DataService:
    svc = getattr(request.app.state, "data_service", None)
    if svc is None:
        raise RuntimeError("DataService not initialized")
    return svc


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    # Convert NaN/NaT to None for JSON serialization.
    cleaned = df.where(pd.notnull(df), None)
    return cleaned.to_dict(orient="records")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest, request: Request) -> PredictionResponse:
    prediction_service = get_prediction_service(request)
    try:
        result = prediction_service.predict_latest(symbol=req.symbol, period=req.period)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        symbol=result.symbol,
        probabilities=result.probabilities,
        system_score=result.system_score,
        tech_score=result.tech_score,
        fund_score=result.fund_score,
        proba_buy=result.proba_buy,
        recommendation=result.recommendation,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/stock/{symbol}/data", response_model=StockDataResponse)
def stock_data(symbol: str, request: Request, period: str = "10y") -> StockDataResponse:
    data_service = get_data_service(request)
    try:
        df = data_service.fetch_stock_data(symbol=symbol, period=period)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {e}")

    payload = _df_to_records(df)
    return StockDataResponse(symbol=symbol, period=period, rows=len(payload), data=payload)


@router.get("/stock/{symbol}/indicators", response_model=IndicatorsResponse)
def indicators(symbol: str, request: Request, period: str = "10y") -> IndicatorsResponse:
    data_service = get_data_service(request)
    try:
        df = data_service.fetch_stock_data(symbol=symbol, period=period)
        df = data_service.add_indicators(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indicator calculation failed: {e}")

    payload = _df_to_records(df)
    return IndicatorsResponse(symbol=symbol, period=period, rows=len(payload), data=payload)

