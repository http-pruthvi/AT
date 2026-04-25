import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "7860"))
    server_url: str = os.getenv("SERVER_URL", "http://localhost:7860")
    env_url: str = os.getenv("ADAPTIVE_TUTOR_ENV_URL", "https://http-pruthvi-adaptive-tutor-ai.hf.space")

    ai_model_name: str = os.getenv("AI_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    trained_model_path: str = os.getenv("TRAINED_MODEL_PATH", str(ROOT_DIR / "adaptive_tutor_trained"))
    lazy_load_ai: bool = _as_bool("LAZY_LOAD_AI", True)

    profiles_dir: str = os.getenv("PROFILES_DIR", str(ROOT_DIR / "profiles"))
    logs_dir: str = os.getenv("LOGS_DIR", str(ROOT_DIR / "logs"))
    outputs_dir: str = os.getenv("OUTPUTS_DIR", str(ROOT_DIR / "outputs"))

    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

    api_token: str = os.getenv("API_AUTH_TOKEN", "")
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
    privacy_redact_answers: bool = _as_bool("PRIVACY_REDACT_ANSWERS", False)

    model_registry_dir: str = os.getenv("MODEL_REGISTRY_DIR", str(ROOT_DIR / "model_registry"))
    min_eval_score: float = float(os.getenv("MIN_EVAL_SCORE", "0.60"))


settings = Settings()

