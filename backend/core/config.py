from __future__ import annotations

from environment import config as env_config


class Settings:
    HOST: str = env_config.BACKEND_HOST
    PORT: int = env_config.BACKEND_PORT
    DATABASE_URL: str = env_config.DATABASE_URL
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
    ]


settings = Settings()
