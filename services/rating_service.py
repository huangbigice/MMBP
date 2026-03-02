"""
股票品質評級服務。

根據回測 Sharpe Ratio 對股票進行 A-F 評級，並提供替代建議。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from services.backtest_service import BacktestService


@dataclass(frozen=True)
class QualityTier:
    """品質等級定義。"""

    rating: str
    min_sharpe: float
    label: str
    color: str
    portfolio_eligible: bool
    description: str


@dataclass(frozen=True)
class StockRating:
    """股票評級結果。"""

    symbol: str
    rating: str
    label: str
    color: str
    sharpe_ratio: float | None
    portfolio_eligible: bool
    warning: bool
    description: str
    alternatives: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class QualityStock:
    """預設優質股票資訊。"""

    symbol: str
    name: str
    category: str
    expected_rating: str


class RatingService:
    """
    股票品質評級服務。

    基於回測 Sharpe Ratio 評估股票品質，提供 A-F 五級評分。
    為低評級股票提供優質替代建議。
    """

    def __init__(
        self,
        backtest_service: BacktestService,
        config_path: str | Path | None = None,
    ):
        """
        初始化評級服務。

        Parameters
        ----------
        backtest_service : BacktestService
            回測服務實例，用於計算 Sharpe Ratio。
        config_path : str | Path | None
            品質評級配置檔路徑。None 時使用預設路徑。
        """
        self._backtest_service = backtest_service

        # 載入配置
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "stock_quality.yaml"
        else:
            config_path = Path(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        # 解析品質等級
        self._tiers = self._parse_tiers()

        # 解析預設優質股票池
        self._quality_stocks = self._parse_quality_stocks()

        # 評級配置
        rating_cfg = self._config.get("rating_config", {})
        self._backtest_years = rating_cfg.get("backtest_years", 3)
        self._use_vol_targeting = rating_cfg.get("use_vol_targeting", True)
        self._max_alternatives = rating_cfg.get("max_alternatives", 3)
        self._warning_ratings = set(rating_cfg.get("warning_ratings", ["D", "F"]))

    def _parse_tiers(self) -> list[QualityTier]:
        """解析品質等級配置。"""
        tiers_config = self._config.get("quality_tiers", {})
        tiers = []

        for rating in ["A", "B", "C", "D", "F"]:
            tier_cfg = tiers_config.get(rating, {})
            tiers.append(
                QualityTier(
                    rating=rating,
                    min_sharpe=tier_cfg.get("min_sharpe", -999.0),
                    label=tier_cfg.get("label", rating),
                    color=tier_cfg.get("color", "gray"),
                    portfolio_eligible=tier_cfg.get("portfolio_eligible", False),
                    description=tier_cfg.get("description", ""),
                )
            )

        # 按 min_sharpe 降序排序（從高到低）
        tiers.sort(key=lambda t: t.min_sharpe, reverse=True)
        return tiers

    def _parse_quality_stocks(self) -> list[QualityStock]:
        """解析預設優質股票池。"""
        stocks_config = self._config.get("default_quality_stocks", [])
        return [
            QualityStock(
                symbol=s.get("symbol", ""),
                name=s.get("name", ""),
                category=s.get("category", ""),
                expected_rating=s.get("expected_rating", ""),
            )
            for s in stocks_config
        ]

    def _determine_rating(self, sharpe: float | None) -> QualityTier:
        """
        根據 Sharpe Ratio 判定評級。

        Parameters
        ----------
        sharpe : float | None
            Sharpe Ratio。None 時視為 F 級。

        Returns
        -------
        QualityTier
            對應的品質等級。
        """
        if sharpe is None:
            # 無法計算 Sharpe（例如資料不足），給予 F 級
            return self._tiers[-1]  # F 級在最後

        # 從高到低比對門檻
        for tier in self._tiers:
            if sharpe >= tier.min_sharpe:
                return tier

        # 理論上不會到這裡（F 級 min_sharpe 為極小值）
        return self._tiers[-1]

    def calculate_stock_rating(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """
        計算股票品質評級。

        Parameters
        ----------
        symbol : str
            股票代碼，如 "2330.TW"。
        start : str | None
            回測開始日期 YYYY-MM-DD。None 時使用最近 N 年。
        end : str | None
            回測結束日期 YYYY-MM-DD。None 時使用今天。

        Returns
        -------
        dict[str, Any]
            評級結果，包含 rating, label, color, sharpe_ratio, portfolio_eligible, warning, alternatives。
        """
        # 確定回測期間
        if end is None:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(end, "%Y-%m-%d")

        if start is None:
            start_date = end_date - timedelta(days=365 * self._backtest_years)
            start = start_date.strftime("%Y-%m-%d")

        # 執行回測取得 Sharpe Ratio
        try:
            backtest_result = self._backtest_service.run_backtest(
                symbol=symbol,
                start=start,
                end=end if isinstance(end, str) else end_date.strftime("%Y-%m-%d"),
                use_vol_targeting=self._use_vol_targeting,
            )
            sharpe = backtest_result.sharpe_ratio
        except Exception as e:
            # 回測失敗（例如資料不足、股票不存在），視為無法評級
            print(f"評級失敗 {symbol}: {e}")
            sharpe = None

        # 判定評級
        tier = self._determine_rating(sharpe)

        # 是否需要警告
        warning = tier.rating in self._warning_ratings

        # 若為低評級，提供替代建議
        alternatives = None
        if warning:
            alternatives = self._suggest_alternatives(symbol, tier.rating)

        return {
            "symbol": symbol,
            "rating": tier.rating,
            "label": tier.label,
            "color": tier.color,
            "sharpe_ratio": sharpe,
            "portfolio_eligible": tier.portfolio_eligible,
            "warning": warning,
            "description": tier.description,
            "alternatives": alternatives,
        }

    def _suggest_alternatives(
        self,
        symbol: str,
        rating: str,
    ) -> list[dict[str, Any]]:
        """
        為低評級股票提供優質替代建議。

        Parameters
        ----------
        symbol : str
            原股票代碼。
        rating : str
            原股票評級。

        Returns
        -------
        list[dict[str, Any]]
            替代股票列表，每項包含 symbol, name, category, rating, sharpe。
        """
        alternatives = []

        # 從預設優質股票池中選擇
        for stock in self._quality_stocks[: self._max_alternatives]:
            # 跳過相同股票
            if stock.symbol == symbol:
                continue

            # 簡化版：直接使用預期評級（避免重複回測耗時）
            # 實際生產環境可考慮快取評級結果
            alternatives.append(
                {
                    "symbol": stock.symbol,
                    "name": stock.name,
                    "category": stock.category,
                    "rating": stock.expected_rating,
                    "sharpe": None,  # 可選：實際計算 Sharpe
                }
            )

            if len(alternatives) >= self._max_alternatives:
                break

        return alternatives

    def calculate_batch_ratings(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        批次計算多檔股票評級。

        Parameters
        ----------
        symbols : list[str]
            股票代碼列表。
        start : str | None
            回測開始日期。
        end : str | None
            回測結束日期。

        Returns
        -------
        dict[str, dict[str, Any]]
            評級結果字典，key 為 symbol，value 為評級結果。
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.calculate_stock_rating(symbol, start, end)
            except Exception as e:
                print(f"批次評級失敗 {symbol}: {e}")
                # 失敗時給予 F 級
                results[symbol] = {
                    "symbol": symbol,
                    "rating": "F",
                    "label": "評級失敗",
                    "color": "red",
                    "sharpe_ratio": None,
                    "portfolio_eligible": False,
                    "warning": True,
                    "description": f"評級計算失敗: {e}",
                    "alternatives": None,
                }

        return results

    def filter_eligible_stocks(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
    ) -> list[tuple[str, float]]:
        """
        篩選符合組合資格的股票（A/B 級）。

        Parameters
        ----------
        symbols : list[str]
            候選股票代碼列表。
        start : str | None
            回測開始日期。
        end : str | None
            回測結束日期。

        Returns
        -------
        list[tuple[str, float]]
            符合資格的股票列表，每項為 (symbol, sharpe_ratio)。
            按 Sharpe Ratio 降序排序。
        """
        ratings = self.calculate_batch_ratings(symbols, start, end)

        # 篩選符合組合資格的股票
        eligible = []
        for symbol, rating_info in ratings.items():
            if rating_info["portfolio_eligible"] and rating_info["sharpe_ratio"] is not None:
                eligible.append((symbol, rating_info["sharpe_ratio"]))

        # 按 Sharpe Ratio 降序排序
        eligible.sort(key=lambda x: x[1], reverse=True)

        return eligible

    def get_quality_stocks(self) -> list[QualityStock]:
        """取得預設優質股票池。"""
        return self._quality_stocks.copy()
