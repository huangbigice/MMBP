from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router as api_router
from audit import AuditLogger
from audit.middleware import AuditMiddleware
from config.config_loader import ConfigLoader
from config.versioning import get_model_version_info
from logger.logger_manager import LoggerManager
from models.model_loader import ModelLoadConfig, ModelLoader
from services.backtest_service import BacktestService
from services.data_service import DataService
from services.prediction_service import PredictionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Init shared dependencies ----
    config = ConfigLoader().config
    logger = LoggerManager().get_logger()
    logger.info("=== make_money API starting ===")

    model_version_info = get_model_version_info()
    model_loader = ModelLoader(ModelLoadConfig(model_path=config.model_path))
    data_service = DataService()
    prediction_service = PredictionService(
        model_loader=model_loader,
        data_service=data_service,
        model_version_info=model_version_info,
    )
    backtest_service = BacktestService(
        data_service=data_service,
        model_loader=model_loader,
        model_version_info=model_version_info,
    )

    removed = audit_logger.cleanup_expired()
    if removed:
        logger.info("Audit log cleanup: removed %d expired file(s)", removed)

    app.state.logger = logger
    app.state.model_version_info = model_version_info
    app.state.model_loader = model_loader
    app.state.data_service = data_service
    app.state.prediction_service = prediction_service
    app.state.backtest_service = backtest_service
    app.state.audit_logger = audit_logger

    yield

    logger.info("=== make_money API shutdown ===")


config = ConfigLoader().config
audit_logger = AuditLogger(
    log_dir=config.audit_log_dir,
    retention_days=config.audit_retention_days,
    hmac_secret=config.audit_hmac_secret,
)

app = FastAPI(title="make_money API", version="0.1.0", lifespan=lifespan)

app.add_middleware(AuditMiddleware, audit_logger=audit_logger)
app.include_router(api_router)
