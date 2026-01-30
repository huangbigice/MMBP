from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

# 預測請求
class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., description="Stock symbol, e.g. 2330.TW")
    period: str = Field("10y", description="yfinance period, e.g. 1y, 5y, 10y")

# 預測回應
class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    probabilities: Dict[str, float]
    system_score: float
    tech_score: float
    fund_score: float
    proba_buy: float
    recommendation: str
    timestamp: datetime

# 股票資料
class StockDataResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    period: str
    rows: int
    data: list[dict[str, Any]]

# 技術指標
class IndicatorsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    period: str
    rows: int
    data: list[dict[str, Any]]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str


class ChatStreamRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., description="Stock symbol for context, e.g. 2330.TW")
    message: str = Field(..., description="User message")
    context: Optional[str] = Field(None, description="Optional extra context text")


# 回測權益曲線單點
class EquityCurvePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    cumulative_return: float


# 回測回應（單資產與組合共用）
class BacktestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    start: str
    end: str
    annualized_return: float
    volatility: float
    max_drawdown: float
    trade_count: int
    sharpe_ratio: Optional[float] = None
    equity_curve: list[EquityCurvePoint]


# 組合回測請求
class PortfolioBacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbols: list[str] = Field(..., min_length=1, description="標的代碼列表，如 2330.TW, 2454.TW")
    start: Optional[str] = Field(None, description="開始日期 YYYY-MM-DD")
    end: Optional[str] = Field(None, description="結束日期 YYYY-MM-DD")
    target_vol_annual: Optional[float] = Field(0.10, description="單資產目標年化波動率")
    vol_lookback: int = Field(20, description="波動率估計滾動天數")
    max_leverage: float = Field(1.0, description="單資產最大槓桿")
    target_portfolio_vol_annual: Optional[float] = Field(0.10, description="組合層級目標年化波動率")
    max_single_weight: float = Field(0.40, description="單一資產權重上限")

