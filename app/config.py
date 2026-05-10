from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Agentic Judge"
    database_url: str = "sqlite+aiosqlite:///./agentic_judge.db"
    secret_key: str = "change-me-in-production-use-a-real-secret"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 hours
    max_judge_iterations: int = 5
    generator_model: str = "gpt-4"
    judge_timeout_seconds: int = 30
    ws_heartbeat_interval: int = 5

    model_config = {"env_prefix": "AJ_"}


settings = Settings()
