from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.schemas import (
    AlternativeStock,
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
    StockQuoteResponse,
    StockRatingResponse,
    StressTestRequest,
    StressTestResponse,
    WalkForwardCompareRequest,
    WalkForwardCompareResponse,
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


def get_rating_service(request: Request):
    """取得評級服務（從 app.state 或即時建立）。"""
    svc = getattr(request.app.state, "rating_service", None)
    if svc is None:
        # 若尚未初始化，使用 backtest_service 即時建立
        from services.rating_service import RatingService
        backtest_service = get_backtest_service(request)
        svc = RatingService(backtest_service=backtest_service)
        request.app.state.rating_service = svc
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
def stock_data(
    symbol: str,
    request: Request,
    period: str = "10y",
    interval: str | None = None,
) -> StockDataResponse:
    data_service = get_data_service(request)
    try:
        effective_period = period
        if interval is not None and period == "10y":
            # Default intraday windows (yfinance limitations):
            # - 1m/10m: typically <= 7d
            # - others: up to 60d
            effective_period = "7d" if interval in ("1m", "10m") else "60d"
        df = data_service.fetch_stock_data(symbol=symbol, period=effective_period, interval=interval)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {e}")

    try:
        quote = data_service.get_quote(symbol=symbol)
        df = data_service.apply_quote_to_last_row(df, quote)
    except Exception:
        pass  # 報價失敗則沿用原 df

    payload = _df_to_records(df)
    return StockDataResponse(symbol=symbol, period=effective_period, rows=len(payload), data=payload)


@router.get("/stock/{symbol}/quote", response_model=StockQuoteResponse)
def stock_quote(symbol: str, request: Request) -> StockQuoteResponse:
    """取得即時／延遲報價（yfinance）。"""
    data_service = get_data_service(request)
    try:
        result = data_service.get_quote(symbol=symbol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quote fetch failed: {e}")
    return StockQuoteResponse(
        symbol=result["symbol"],
        current_price=result["current_price"],
        previous_close=result["previous_close"],
        change=result["change"],
        change_percent=result["change_percent"],
    )


@router.get("/stock/{symbol}/indicators", response_model=IndicatorsResponse)
def indicators(
    symbol: str,
    request: Request,
    period: str = "10y",
    interval: str | None = None,
) -> IndicatorsResponse:
    data_service = get_data_service(request)
    try:
        effective_period = period
        if interval is not None and period == "10y":
            effective_period = "7d" if interval in ("1m", "10m") else "60d"
        df = data_service.fetch_stock_data(symbol=symbol, period=effective_period, interval=interval)
        # Intraday first version: return base OHLCV only (frontend computes chart indicators)
        if interval is None:
            df = data_service.add_indicators(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indicator calculation failed: {e}")

    try:
        quote = data_service.get_quote(symbol=symbol)
        df = data_service.apply_quote_to_last_row(df, quote)
        if interval is None:
            df = data_service.add_indicators(df)
    except Exception:
        pass  # 報價失敗則沿用原 df，不重算指標

    payload = _df_to_records(df)
    return IndicatorsResponse(symbol=symbol, period=effective_period, rows=len(payload), data=payload)


@router.get("/stock/{symbol}/rating", response_model=StockRatingResponse)
def stock_rating(
    symbol: str,
    request: Request,
    start: str | None = None,
    end: str | None = None,
) -> StockRatingResponse:
    """
    取得股票品質評級與組合資格。
    
    根據回測 Sharpe Ratio 評估股票品質，提供 A-F 五級評分。
    低評級股票會顯示警告並提供優質替代建議。
    
    Parameters
    ----------
    symbol : str
        股票代碼，如 2330.TW。
    start : str | None
        回測開始日期 YYYY-MM-DD。None 時使用最近 3 年。
    end : str | None
        回測結束日期 YYYY-MM-DD。None 時使用今天。
    """
    rating_service = get_rating_service(request)
    try:
        result = rating_service.calculate_stock_rating(
            symbol=symbol,
            start=start,
            end=end,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"評級計算失敗: {e}")

    # 轉換替代建議格式
    alternatives = None
    if result.get("alternatives"):
        alternatives = [
            AlternativeStock(
                symbol=alt["symbol"],
                name=alt["name"],
                category=alt["category"],
                rating=alt["rating"],
                sharpe=alt.get("sharpe"),
            )
            for alt in result["alternatives"]
        ]

    return StockRatingResponse(
        symbol=result["symbol"],
        rating=result["rating"],
        label=result["label"],
        color=result["color"],
        sharpe_ratio=result["sharpe_ratio"],
        portfolio_eligible=result["portfolio_eligible"],
        warning=result["warning"],
        description=result["description"],
        alternatives=alternatives,
    )


@router.get("/backtest", response_model=BacktestResponse)
def backtest(
    request: Request,
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    use_vol_targeting: bool = True,
    target_vol_annual: float | None = None,
    vol_lookback: int | None = None,
    max_leverage: float | None = None,
    momentum_filter: bool = False,
) -> BacktestResponse:
    """
    單資產回測，支援波動率目標倉位與動量濾鏡。
    
    Parameters
    ----------
    target_vol_annual : float | None
        目標年化波動率。None 時使用配置預設值（18%）。
    momentum_filter : bool
        是否啟用動量濾鏡（return_5 > 0）。預設為 False。
    """
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
            momentum_filter=momentum_filter,
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


@router.post("/backtest/stress-test", response_model=StressTestResponse)
def stress_test_backtest(
    request: Request,
    body: StressTestRequest,
) -> StressTestResponse:
    """
    對回測結果執行多時期壓力測試。
    
    測試策略在歷史危機時期（如 2022 熊市、2020 COVID 崩盤等）的表現。
    """
    from services.backtest import stress_test_multiple_periods
    
    backtest_service = get_backtest_service(request)
    
    try:
        # 執行完整回測以獲取詳細 DataFrame
        df = backtest_service.get_backtest_df(
            symbol=body.symbol,
            start=body.start,
            end=body.end,
            momentum_filter=body.momentum_filter,
        )
        
        # 執行多時期壓力測試
        scenarios = body.scenarios if body.scenarios else None
        results_df = stress_test_multiple_periods(
            df=df,
            scenarios=scenarios,
            risk_free_rate=0.02,
        )
        
        # 轉換為回應格式
        scenarios_results = results_df.to_dict(orient="records")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"壓力測試失敗: {e}")
    
    return StressTestResponse(
        symbol=body.symbol,
        start=body.start or "N/A",
        end=body.end or "N/A",
        scenarios=scenarios_results,
    )


@router.post("/backtest/walk-forward-compare", response_model=WalkForwardCompareResponse)
def walk_forward_compare(
    request: Request,
    body: WalkForwardCompareRequest,
) -> WalkForwardCompareResponse:
    """
    比較不同策略變體的 Walk-Forward 穩定性。
    
    測試四種策略：
    1. 基準策略（無動量濾鏡）
    2. 動量濾鏡策略（return_5 > 0）
    3. 多資產策略（若提供 portfolio_symbols）
    4. 動量 + 多資產策略（若提供 portfolio_symbols）
    """
    from services.backtest import compare_walk_forward_strategies
    
    backtest_service = get_backtest_service(request)
    
    try:
        # 執行策略比較
        comparison = compare_walk_forward_strategies(
            symbol=body.symbol,
            start=body.start,
            end=body.end,
            backtest_service=backtest_service,
            train_years=body.train_years,
            test_years=body.test_years,
            portfolio_symbols=body.portfolio_symbols if body.include_portfolio else None,
        )
        
        # 轉換摘要表為字典
        summary_records = comparison["summary"].to_dict(orient="records")
        
        # 提取各策略的詳細結果
        strategies = {}
        for strategy_name in ["baseline", "momentum", "portfolio", "momentum_portfolio"]:
            if strategy_name in comparison:
                analysis = comparison[strategy_name]
                if "note" not in analysis:  # 跳過未實現的策略
                    strategies[strategy_name] = {
                        "n_windows": analysis["n_windows"],
                        "sharpe_mean": analysis["sharpe_mean"],
                        "sharpe_std": analysis["sharpe_std"],
                        "negative_sharpe_pct": analysis["negative_sharpe_pct"],
                        "avg_return": analysis["avg_return"],
                        "avg_mdd": analysis["avg_mdd"],
                        "avg_volatility": analysis["avg_volatility"],
                    }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Walk-Forward 比較失敗: {e}")
    
    return WalkForwardCompareResponse(
        symbol=body.symbol,
        start=body.start,
        end=body.end,
        train_years=body.train_years,
        test_years=body.test_years,
        strategies=strategies,
        summary=summary_records,
    )

