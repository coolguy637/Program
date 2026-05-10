import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import engine
from app.models import Base
from app.routers import api_keys, judge, override, simulator, users

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

app = FastAPI(
    title=settings.app_name,
    description=(
        "Advanced Agentic Judge & Autonomous Operator platform. "
        "Manages an internal Conflict Loop between a Generator and a panel of expert judges "
        "to bridge the gap between raw AI output and high-end human standards."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(users.router)
app.include_router(api_keys.router)
app.include_router(judge.router)
app.include_router(override.router)
app.include_router(simulator.router)

# Static files (dashboard)
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")


@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logging.getLogger(__name__).info("Agentic Judge platform started")
