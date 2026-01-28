from __future__ import annotations

import logging


class LoggerManager:
    """
    Simple logger factory.

    This keeps the original architecture (LoggerManager) but uses stdlib logging.
    """

    def __init__(self, name: str = "make_money") -> None:
        self._name = name

    def get_logger(self) -> logging.Logger:
        logger = logging.getLogger(self._name)
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger

