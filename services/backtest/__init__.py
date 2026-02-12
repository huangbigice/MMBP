"""
回測子模組：多因子 regime、多空 ATR 進出場、Walk-forward、壓力測試。

- regime: 多因子趨勢狀態 (bull / bear / neutral)
- position_engine: 多空對稱 ATR 停損進出場，輸出 position -1 / 0 / 1
- walk_forward: Rolling train/test 框架
- stress: Monte Carlo / block bootstrap / 單日衝擊
"""

from .position_engine import compute_position_long_short
from .regime import add_adx, compute_regime

__all__ = [
    "add_adx",
    "compute_regime",
    "compute_position_long_short",
]
