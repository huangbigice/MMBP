from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.schemas import (
    BacktestResponse,
    ChatStreamRequest,
    EquityCurvePoint,
    HealthResponse,
    IndicatorsResponse,
    ModelInfoResponse,
    PortfolioBacktestRequest,
    PredictionRequest,
    PredictionResponse,
    StockDataResponse,
)
from models.ollama_loader import OllamaLoader
from services.backtest_service import BacktestService
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


def get_backtest_service(request: Request) -> BacktestService:
    svc = getattr(request.app.state, "backtest_service", None)
    if svc is None:
        raise RuntimeError("BacktestService not initialized")
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


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    """目前使用之模型／策略版本與上線日，供稽核與前端顯示。"""
    info = getattr(request.app.state, "model_version_info", None)
    if info is None:
        raise RuntimeError("model_version_info not initialized")
    return ModelInfoResponse(
        model_version=info.model_version,
        strategy_version=info.strategy_version,
        model_effective_date=info.model_effective_date,
        training_interval=info.training_interval,
        assumptions=list(info.assumptions),
    )

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
        model_version=result.model_version,
        strategy_version=result.strategy_version,
        model_effective_date=result.model_effective_date,
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


@router.get("/backtest", response_model=BacktestResponse)
def backtest(
    request: Request,
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    use_vol_targeting: bool = True,
    target_vol_annual: float = 0.10,
    vol_lookback: int = 20,
    max_leverage: float = 1.0,
) -> BacktestResponse:
    """單資產回測，支援波動率目標倉位（預設開啟）。"""
    backtest_service = get_backtest_service(request)
    try:
        result = backtest_service.run_backtest(
            symbol=symbol,
            start=start,
            end=end,
            use_vol_targeting=use_vol_targeting,
            target_vol_annual=target_vol_annual,
            vol_lookback=vol_lookback,
            max_leverage=max_leverage,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")

    return BacktestResponse(
        symbol=result.symbol,
        start=result.start,
        end=result.end,
        annualized_return=result.annualized_return,
        volatility=result.volatility,
        max_drawdown=result.max_drawdown,
        trade_count=result.trade_count,
        sharpe_ratio=result.sharpe_ratio,
        equity_curve=[
            EquityCurvePoint(date=p["date"], cumulative_return=p["cumulative_return"])
            for p in result.equity_curve
        ],
        model_version=result.model_version,
        strategy_version=result.strategy_version,
        model_effective_date=result.model_effective_date,
    )


@router.post("/backtest/portfolio", response_model=BacktestResponse)
def backtest_portfolio(
    request: Request,
    body: PortfolioBacktestRequest,
) -> BacktestResponse:
    """多資產組合回測：逆波動率權重 + 組合層級波動率目標與單一資產權重上限。"""
    backtest_service = get_backtest_service(request)
    try:
        result = backtest_service.run_portfolio_backtest(
            symbols=body.symbols,
            start=body.start,
            end=body.end,
            target_vol_annual=body.target_vol_annual,
            vol_lookback=body.vol_lookback,
            max_leverage=body.max_leverage,
            target_portfolio_vol_annual=body.target_portfolio_vol_annual,
            max_single_weight=body.max_single_weight,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio backtest failed: {e}")

    return BacktestResponse(
        symbol=result.symbol,
        start=result.start,
        end=result.end,
        annualized_return=result.annualized_return,
        volatility=result.volatility,
        max_drawdown=result.max_drawdown,
        trade_count=result.trade_count,
        sharpe_ratio=result.sharpe_ratio,
        equity_curve=[
            EquityCurvePoint(date=p["date"], cumulative_return=p["cumulative_return"])
            for p in result.equity_curve
        ],
        model_version=result.model_version,
        strategy_version=result.strategy_version,
        model_effective_date=result.model_effective_date,
    )

