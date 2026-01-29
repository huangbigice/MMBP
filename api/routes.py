from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.schemas import (
    ChatStreamRequest,
    HealthResponse,
    IndicatorsResponse,
    PredictionRequest,
    PredictionResponse,
    StockDataResponse,
)
from models.ollama_loader import OllamaLoader
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

def _sse_pack_data(data: str) -> str:
    """
    Pack a chunk into a minimal SSE event.

    SSE 'data:' cannot safely contain raw newlines unless split by lines.
    Here we split into multiple data lines to preserve newlines.
    """
    lines = data.splitlines() or [""]
    return "".join([f"data: {line}\n" for line in lines]) + "\n"

def _sse_event(event: str, data: str) -> str:
    lines = data.splitlines() or [""]
    return f"event: {event}\n" + "".join([f"data: {line}\n" for line in lines]) + "\n"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()

@router.post("/chat/stream")
async def chat_stream(req: ChatStreamRequest) -> StreamingResponse:
    """
    Stream assistant response via SSE (text/event-stream).
    """
    loader = OllamaLoader()

    # Keep prompt small and deterministic; rely on context text from frontend.
    system_prompt = (
        "你是投資分析助理。回覆請用繁體中文，內容聚焦在使用者問題與提供的技術指標摘要。"
        "避免提供保證獲利或確定性承諾；可用風險提示。"
    )

    async def gen():
        try:
            async for chunk in loader.stream_chat(
                user_message=req.message,
                system_prompt=system_prompt,
                context=f"股票代碼: {req.symbol}\n{req.context or ''}".strip(),
            ):
                yield _sse_pack_data(chunk)
            yield _sse_event("done", "[DONE]")
        except Exception as e:
            # Let client show an error bubble.
            yield _sse_event("error", f"{e}")

    return StreamingResponse(gen(), media_type="text/event-stream")


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

