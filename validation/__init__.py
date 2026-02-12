"""
五層 Alpha 驗證框架。

- Layer 1: Null Model 殺戮測試（真實 vs 打亂 vs 永遠觀望）
- Layer 2: Walk-forward 穩定性（Sharpe 分佈）
- Layer 3: Regime 生存測試（多頭/空頭/高VIX/低VIX）
- Layer 4: Benchmark 智商測試（B&H、Momentum、MA cross、RSI）
- Layer 5: Economic Plausibility（經濟來源自評）
"""

from .five_layer_tests import (
    run_layer1_null_test,
    run_layer2_walk_forward_stability,
    run_layer3_regime_survival,
    run_layer4_benchmark_iq,
    run_layer5_economic_plausibility,
    run_all_five_layers,
)

__all__ = [
    "run_layer1_null_test",
    "run_layer2_walk_forward_stability",
    "run_layer3_regime_survival",
    "run_layer4_benchmark_iq",
    "run_layer5_economic_plausibility",
    "run_all_five_layers",
]
