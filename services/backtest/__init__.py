"""
回測子模組：多因子 regime、多空 ATR 進出場、Walk-forward、壓力測試。

- regime: 多因子趨勢狀態 (bull / bear / neutral)
- position_engine: 多空對稱 ATR 停損進出場，輸出 position -1 / 0 / 1
- walk_forward: Rolling train/test 框架，支援單資產與多資產組合驗證
- stress: Monte Carlo / block bootstrap / 單日衝擊 / 歷史危機時期測試
"""

from .position_engine import compute_position_long_short
from .regime import add_adx, compute_regime
from .stress import (
    CRISIS_SCENARIOS,
    MonteCarloResult,
    analyze_stress_distribution,
    block_bootstrap,
    monte_carlo_bootstrap,
    stress_single_day_shock,
    stress_test_multiple_periods,
    stress_test_period,
)
from .walk_forward import (
    WalkForwardResult,
    WalkForwardWindow,
    compare_walk_forward_strategies,
    generate_walk_forward_windows,
    run_walk_forward,
    run_walk_forward_portfolio,
)

__all__ = [
    "CRISIS_SCENARIOS",
    "MonteCarloResult",
    "WalkForwardResult",
    "WalkForwardWindow",
    "add_adx",
    "analyze_stress_distribution",
    "block_bootstrap",
    "compare_walk_forward_strategies",
    "compute_position_long_short",
    "compute_regime",
    "generate_walk_forward_windows",
    "monte_carlo_bootstrap",
    "run_walk_forward",
    "run_walk_forward_portfolio",
    "stress_single_day_shock",
    "stress_test_multiple_periods",
    "stress_test_period",
]
