"""
Model and strategy versioning for model risk management.

Single source of truth for model_version, strategy_version, and effective date.
Override via env: MODEL_VERSION, STRATEGY_VERSION, MODEL_EFFECTIVE_DATE.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# Defaults; change here when releasing a new version
MODEL_VERSION = "1.0.0"
STRATEGY_VERSION = "1.0.0"
MODEL_EFFECTIVE_DATE = "2026-02-01"

# Optional metadata for docs / model-info API
TRAINING_INTERVAL: Optional[str] = None  # e.g. "2020-01-01 to 2025-12-31"
MAIN_ASSUMPTIONS: tuple[str, ...] = (
    "Random Forest 三類標籤：不建議持有 / 長期持有 / 觀望",
    "系統評分權重：模型機率 0.45、基本面 0.30、技術面 0.25",
    "回測：2 ATR 移動停損、波動率目標倉位、手續費 0.15%",
)


@dataclass(frozen=True)
class ModelVersionInfo:
    """Current model and strategy version info for API and audit."""

    model_version: str
    strategy_version: str
    model_effective_date: str
    training_interval: Optional[str] = None
    assumptions: tuple[str, ...] = ()


def get_model_version_info() -> ModelVersionInfo:
    """
    Return current model/strategy version info.
    Env vars MODEL_VERSION, STRATEGY_VERSION, MODEL_EFFECTIVE_DATE override defaults.
    """
    model_version = os.getenv("MODEL_VERSION", MODEL_VERSION)
    strategy_version = os.getenv("STRATEGY_VERSION", STRATEGY_VERSION)
    model_effective_date = os.getenv("MODEL_EFFECTIVE_DATE", MODEL_EFFECTIVE_DATE)
    training_interval = os.getenv("TRAINING_INTERVAL", TRAINING_INTERVAL)
    return ModelVersionInfo(
        model_version=model_version,
        strategy_version=strategy_version,
        model_effective_date=model_effective_date,
        training_interval=training_interval,
        assumptions=MAIN_ASSUMPTIONS,
    )
