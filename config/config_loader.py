from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
AUDIT_LOG_DIR = os.getenv("AUDIT_LOG_DIR")
AUDIT_RETENTION_DAYS = os.getenv("AUDIT_RETENTION_DAYS")
AUDIT_HMAC_SECRET = os.getenv("AUDIT_HMAC_SECRET")


@dataclass(frozen=True)
class AppConfig:
    """
    Centralized runtime configuration.

    Keep it minimal for now; expand as the backend grows.
    """

    model_path: Path
    audit_log_dir: Path
    audit_retention_days: int
    audit_hmac_secret: str


class ConfigLoader:
    """
    Loads configuration from environment variables / defaults.
    """

    def __init__(self) -> None:
        default_model_path = Path(__file__).resolve().parents[1] / "rf_model_2330.pkl"
        model_path = Path(os.getenv("MODEL_PATH", str(default_model_path))).expanduser()

        default_audit_dir = Path(__file__).resolve().parents[1] / "logs" / "audit"
        audit_log_dir = Path(
            os.getenv("AUDIT_LOG_DIR", str(default_audit_dir))
        ).expanduser().resolve()

        audit_retention_days = 180
        if AUDIT_RETENTION_DAYS is not None:
            try:
                audit_retention_days = int(AUDIT_RETENTION_DAYS)
            except ValueError:
                pass

        audit_hmac_secret = os.getenv("AUDIT_HMAC_SECRET", "")

        self._config = AppConfig(
            model_path=model_path,
            audit_log_dir=audit_log_dir,
            audit_retention_days=audit_retention_days,
            audit_hmac_secret=audit_hmac_secret,
        )

    @property
    def config(self) -> AppConfig:
        return self._config

