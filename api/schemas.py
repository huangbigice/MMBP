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

