"""
風險與倉位管理模組。

- position_sizing: 波動率目標倉位 (Volatility Targeting)
- portfolio: 多資產組合風控與權重聚合
"""

from .position_sizing import (
    VolTargetConfig,
    compute_confidence_vol_weights,
    compute_vol_target_weights,
)
from .portfolio import (
    PortfolioRiskConfig,
    align_returns,
    aggregate_portfolio_returns,
    compute_inverse_vol_weights,
)

__all__ = [
    "VolTargetConfig",
    "compute_confidence_vol_weights",
    "compute_vol_target_weights",
    "PortfolioRiskConfig",
    "align_returns",
    "compute_inverse_vol_weights",
    "aggregate_portfolio_returns",
]
