from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
AUDIT_LOG_DIR = os.getenv("AUDIT_LOG_DIR")
AUDIT_RETENTION_DAYS = os.getenv("AUDIT_RETENTION_DAYS")
AUDIT_HMAC_SECRET = os.getenv("AUDIT_HMAC_SECRET")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyParams:
    """策略核心參數。"""
    
    model_threshold: float
    fund_score_threshold: float
    atr_stop_multiplier: float


@dataclass(frozen=True)
class RegimeParams:
    """Regime 分類參數。"""
    
    adx_threshold: float
    adx_period: int
    ma_period: int
    slope_window: int


@dataclass(frozen=True)
class VolatilityTargetingParams:
    """波動率目標參數。"""
    
    default_target_annual: float
    vol_lookback: int
    max_leverage: float
    min_vol_floor: float


@dataclass(frozen=True)
class PerformanceTargets:
    """預期報酬與風險目標。"""
    
    expected_return_min: float
    expected_return_max: float
    max_drawdown_target: float


@dataclass(frozen=True)
class BacktestParams:
    """回測通用參數。"""
    
    risk_free_rate: float
    fee_rate: float


@dataclass(frozen=True)
class MomentumFilterParams:
    """動量濾鏡參數。"""
    
    enabled: bool
    lookback_days: int
    threshold: float


@dataclass(frozen=True)
class PortfolioParams:
    """組合風險管理參數。"""
    
    target_portfolio_vol_annual: float
    max_single_weight: float
    weight_method: str


@dataclass(frozen=True)
class WalkForwardParams:
    """Walk-Forward 驗證參數。"""
    
    train_years: int
    test_years: int
    sharpe_std_threshold: float
    max_negative_sharpe_pct: float


@dataclass(frozen=True)
class MonteCarloParams:
    """Monte Carlo 壓力測試參數。"""
    
    n_simulations: int
    block_size: int
    seed: int | None
    mdd_warning_threshold: float


@dataclass(frozen=True)
class LoggingParams:
    """日誌與除錯參數。"""
    
    log_config_load: bool
    log_walk_forward_windows: bool
    log_stress_test_details: bool


@dataclass(frozen=True)
class StrategyConfig:
    """
    完整策略配置，對應 strategy_config.yaml 結構。
    
    包含所有策略參數的型別安全容器。
    """
    
    strategy: StrategyParams
    regime: RegimeParams
    volatility_targeting: VolatilityTargetingParams
    performance_targets: PerformanceTargets
    backtest: BacktestParams
    momentum_filter: MomentumFilterParams
    portfolio: PortfolioParams
    walk_forward: WalkForwardParams
    monte_carlo: MonteCarloParams
    logging: LoggingParams


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

    @staticmethod
    def load_strategy_config(config_path: Path | None = None) -> StrategyConfig:
        """
        載入策略配置檔（YAML）。
        
        Parameters
        ----------
        config_path : Path | None
            配置檔路徑。若為 None，使用預設路徑 config/strategy_config.yaml。
        
        Returns
        -------
        StrategyConfig
            解析後的策略配置物件。
        
        Raises
        ------
        FileNotFoundError
            配置檔不存在。
        ValueError
            配置檔格式錯誤或缺少必要欄位。
        
        Examples
        --------
        >>> loader = ConfigLoader()
        >>> strategy_config = loader.load_strategy_config()
        >>> print(strategy_config.regime.adx_threshold)
        15.0
        """
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "strategy_config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"策略配置檔不存在: {config_path}\n"
                f"請確保 config/strategy_config.yaml 已正確建立。"
            )
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 解析失敗: {e}")
        
        if data is None:
            raise ValueError("配置檔為空")
        
        # 驗證必要的頂層鍵
        required_sections = [
            "strategy", "regime", "volatility_targeting", "performance_targets",
            "backtest", "momentum_filter", "portfolio", "walk_forward",
            "monte_carlo", "logging"
        ]
        missing = [s for s in required_sections if s not in data]
        if missing:
            raise ValueError(f"配置檔缺少必要區塊: {missing}")
        
        try:
            # 建立各區塊的參數物件
            strategy_config = StrategyConfig(
                strategy=StrategyParams(**data["strategy"]),
                regime=RegimeParams(**data["regime"]),
                volatility_targeting=VolatilityTargetingParams(**data["volatility_targeting"]),
                performance_targets=PerformanceTargets(**data["performance_targets"]),
                backtest=BacktestParams(**data["backtest"]),
                momentum_filter=MomentumFilterParams(**data["momentum_filter"]),
                portfolio=PortfolioParams(**data["portfolio"]),
                walk_forward=WalkForwardParams(**data["walk_forward"]),
                monte_carlo=MonteCarloParams(**data["monte_carlo"]),
                logging=LoggingParams(**data["logging"]),
            )
            
            # 記錄配置載入資訊
            if strategy_config.logging.log_config_load:
                logger.info(f"策略配置載入成功: {config_path}")
                logger.info(f"  - 目標波動率: {strategy_config.volatility_targeting.default_target_annual:.1%}")
                logger.info(f"  - ADX 門檻: {strategy_config.regime.adx_threshold}")
                logger.info(f"  - ATR 停損倍數: {strategy_config.strategy.atr_stop_multiplier}")
                logger.info(f"  - 動量濾鏡預設: {'啟用' if strategy_config.momentum_filter.enabled else '停用'}")
            
            return strategy_config
            
        except TypeError as e:
            raise ValueError(f"配置檔欄位型別錯誤: {e}")
        except KeyError as e:
            raise ValueError(f"配置檔缺少必要欄位: {e}")

