from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

# 版本欄位說明（預測／回測共用）
_DESC_MODEL_VERSION = "模型版本號"
_DESC_STRATEGY_VERSION = "策略／評分邏輯版本號"
_DESC_MODEL_EFFECTIVE_DATE = "模型上線日期 YYYY-MM-DD"

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
    model_version: str = Field(..., description=_DESC_MODEL_VERSION)
    strategy_version: str = Field(..., description=_DESC_STRATEGY_VERSION)
    model_effective_date: str = Field(..., description=_DESC_MODEL_EFFECTIVE_DATE)

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


# 即時報價（yfinance）
class StockQuoteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., description="股票代碼，如 2330.TW")
    current_price: float = Field(..., description="現價")
    previous_close: float = Field(..., description="前收盤價")
    change: float = Field(..., description="漲跌額")
    change_percent: float = Field(..., description="漲跌幅 %")


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"


class ModelInfoResponse(BaseModel):
    """目前使用之模型／策略版本，供稽核與前端顯示。"""

    model_config = ConfigDict(extra="forbid")

    model_version: str = Field(..., description=_DESC_MODEL_VERSION)
    strategy_version: str = Field(..., description=_DESC_STRATEGY_VERSION)
    model_effective_date: str = Field(..., description=_DESC_MODEL_EFFECTIVE_DATE)
    training_interval: Optional[str] = Field(None, description="訓練／驗證區間（若有）")
    assumptions: list[str] = Field(default_factory=list, description="主要假設摘要")


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
    model_version: str = Field(..., description=_DESC_MODEL_VERSION)
    strategy_version: str = Field(..., description=_DESC_STRATEGY_VERSION)
    model_effective_date: str = Field(..., description=_DESC_MODEL_EFFECTIVE_DATE)
    # 品質評級資訊（可選）
    quality_rating: Optional[str] = Field(None, description="品質評級 A-F")
    quality_label: Optional[str] = Field(None, description="評級標籤")
    portfolio_eligible: Optional[bool] = Field(None, description="是否符合組合納入資格")


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


# 壓力測試請求
class StressTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., description="標的代碼，如 2330.TW")
    start: Optional[str] = Field(None, description="回測開始日期 YYYY-MM-DD")
    end: Optional[str] = Field(None, description="回測結束日期 YYYY-MM-DD")
    momentum_filter: bool = Field(False, description="是否啟用動量濾鏡")
    scenarios: Optional[list[str]] = Field(
        None,
        description="測試場景列表，如 ['2022_bear', '2020_covid']。None 時測試所有預設場景"
    )


# 壓力測試回應
class StressTestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    start: str
    end: str
    scenarios: list[dict[str, Any]] = Field(
        ...,
        description="各場景測試結果，包含 scenario, name, start, end, total_return, max_drawdown, sharpe 等"
    )


# Walk-Forward 比較請求
class WalkForwardCompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., description="主要標的代碼，如 2330.TW")
    start: str = Field(..., description="整體區間起日 YYYY-MM-DD")
    end: str = Field(..., description="整體區間迄日 YYYY-MM-DD")
    train_years: int = Field(5, description="每期訓練年數")
    test_years: int = Field(1, description="每期測試年數")
    include_momentum: bool = Field(True, description="是否測試動量濾鏡策略")
    include_portfolio: bool = Field(True, description="是否測試多資產組合策略")
    portfolio_symbols: Optional[list[str]] = Field(
        None,
        description="多資產組合的其他標的，如 ['0050.TW']。None 時只測試單資產策略"
    )


# Walk-Forward 比較回應
class WalkForwardCompareResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    start: str
    end: str
    train_years: int
    test_years: int
    strategies: dict[str, dict[str, Any]] = Field(
        ...,
        description="各策略的統計摘要，鍵值為 baseline, momentum, portfolio, momentum_portfolio"
    )
    summary: list[dict[str, Any]] = Field(
        ...,
        description="策略比較表，包含各策略的 sharpe_mean, sharpe_std, avg_mdd 等指標"
    )


# 股票品質評級
class AlternativeStock(BaseModel):
    """替代股票建議。"""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., description="股票代碼")
    name: str = Field(..., description="股票名稱")
    category: str = Field(..., description="產業類別")
    rating: str = Field(..., description="品質評級 A-F")
    sharpe: Optional[float] = Field(None, description="Sharpe Ratio（若已計算）")


class StockRatingResponse(BaseModel):
    """股票品質評級回應。"""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., description="股票代碼")
    rating: str = Field(..., description="品質評級：A, B, C, D, F")
    label: str = Field(..., description="評級標籤，如「核心持倉」、「避免交易」")
    color: str = Field(..., description="顏色標示：green, blue, yellow, orange, red")
    sharpe_ratio: Optional[float] = Field(None, description="回測 Sharpe Ratio")
    portfolio_eligible: bool = Field(..., description="是否符合組合納入資格（A/B 級為 true）")
    warning: bool = Field(..., description="是否顯示警告（D/F 級為 true）")
    description: str = Field(..., description="評級說明")
    alternatives: Optional[list[AlternativeStock]] = Field(
        None, description="替代建議股票（僅低評級時提供）"
    )

