from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib

import os
from dotenv import load_dotenv
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")

@dataclass(frozen=True)
class ModelLoadConfig:
    """
    Configuration for model loading.

    This project currently uses a single model file (e.g. rf_model_2330.pkl).
    """

    model_path: Path


class ModelLoader:
    """
    A small loader with in-process caching.

    - Keeps the model in memory after first load.
    - Allows injecting a custom config for tests.
    """

    def __init__(self, config: ModelLoadConfig):
        self._config = config
        self._cached_model: Optional[Any] = None

    @property
    def model_path(self) -> Path:
        return self._config.model_path

    def load(self, force_reload: bool = False) -> Any:
        if self._cached_model is not None and not force_reload:
            return self._cached_model

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        model = joblib.load(self.model_path)
        self._cached_model = model
        return model

