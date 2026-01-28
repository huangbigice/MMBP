from __future__ import annotations

from typing import Optional

import pandas as pd


class DuckdbRepository:
    """
    Placeholder repository for future persistence.

    The original codebase imports DuckdbRepository; this stub keeps the interface
    while the API backend currently focuses on prediction endpoints.
    """

    def __init__(self, config=None, logger=None):
        self._config = config
        self._logger = logger

    def save_dataframe(self, df: pd.DataFrame, table_name: str | None = None) -> None:
        # Intentionally no-op for now.
        if self._logger:
            self._logger.info(
                "DuckdbRepository.save_dataframe called (noop). rows=%s table=%s",
                len(df),
                table_name,
            )

    def close(self) -> None:
        # Nothing to close in stub.
        return None

