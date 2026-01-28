from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")

@dataclass(frozen=True)
class AppConfig:
    """
    Centralized runtime configuration.

    Keep it minimal for now; expand as the backend grows.
    """

    model_path: Path


class ConfigLoader:
    """
    Loads configuration from environment variables / defaults.
    """

    def __init__(self) -> None:
        default_model_path = Path(__file__).resolve().parents[1] / "rf_model_2330.pkl"
        model_path = Path(os.getenv("MODEL_PATH", str(default_model_path))).expanduser()
        self._config = AppConfig(model_path=model_path)

    @property
    def config(self) -> AppConfig:
        return self._config

