from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.core.websocket_manager import bus
from backend.models import db
from backend.routers import config as config_router
from backend.routers import experiments, metrics, simulator, training


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    bus.attach_loop(asyncio.get_running_loop())
    yield


app = FastAPI(
    title="Cloud RL Scheduler",
    version="1.0.0",
    description="Dynamic resource allocation in cloud clusters via deep RL.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(training.router)
app.include_router(experiments.router)
app.include_router(simulator.router)
app.include_router(metrics.router)
app.include_router(config_router.router)
