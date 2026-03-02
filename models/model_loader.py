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
    - Supports models saved as dict with "model" and "features" keys.
    """

    def __init__(self, config: ModelLoadConfig):
        self._config = config
        self._cached_model: Optional[Any] = None
        self._cached_features: Optional[list[str]] = None
        self._cached_scaler: Optional[Any] = None

    @property
    def model_path(self) -> Path:
        return self._config.model_path

    def load(self, force_reload: bool = False) -> Any:
        if self._cached_model is not None and not force_reload:
            return self._cached_model

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        loaded = joblib.load(self.model_path)
        
        # 處理模型以字典格式儲存的情況（包含 model、scaler、features）
        # 例如 train_model第六版.py 儲存格式：{"model": model, "scaler": scaler, "features": FEATURES}
        if isinstance(loaded, dict) and "model" in loaded:
            model = loaded["model"]
            # 保存 scaler（如果存在）
            if "scaler" in loaded:
                self._cached_scaler = loaded["scaler"]
            # 保存特徵列表（如果存在）
            if "features" in loaded:
                self._cached_features = loaded["features"]
        else:
            # 向後相容：直接儲存模型物件的情況
            model = loaded
            self._cached_scaler = None
            self._cached_features = None
        
        self._cached_model = model
        return model

    def get_features(self) -> Optional[list[str]]:
        """
        返回模型文件中保存的特徵列表。
        
        如果模型文件是字典格式且包含 "features" 鍵，則返回該列表。
        否則返回 None（表示應使用預設特徵）。
        """
        # 如果尚未載入模型，先載入
        if self._cached_model is None:
            self.load()
        return self._cached_features
    
    def get_scaler(self) -> Optional[Any]:
        """
        返回模型文件中保存的 scaler（StandardScaler）。
        
        如果模型文件是字典格式且包含 "scaler" 鍵，則返回該 scaler。
        否則返回 None（表示模型不需要標準化或使用舊格式）。
        """
        # 如果尚未載入模型，先載入
        if self._cached_model is None:
            self.load()
        return self._cached_scaler

