from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router
from config.config_loader import ConfigLoader
from logger.logger_manager import LoggerManager
from models.model_loader import ModelLoadConfig, ModelLoader
from services.data_service import DataService
from services.prediction_service import PredictionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Init shared dependencies ----
    config = ConfigLoader().config
    logger = LoggerManager().get_logger()
    logger.info("=== make_money API starting ===")

    model_loader = ModelLoader(ModelLoadConfig(model_path=config.model_path))
    data_service = DataService()
    prediction_service = PredictionService(model_loader=model_loader, data_service=data_service)

    app.state.logger = logger
    app.state.model_loader = model_loader
    app.state.data_service = data_service
    app.state.prediction_service = prediction_service

    yield

    logger.info("=== make_money API shutdown ===")


app = FastAPI(title="make_money API", version="0.1.0", lifespan=lifespan)

# CORS (adjust origins as needed for your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
