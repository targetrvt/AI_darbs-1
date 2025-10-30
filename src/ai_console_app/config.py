import os
from typing import Optional

from dotenv import load_dotenv


def load_environment(dotenv_path: Optional[str] = None) -> None:
    load_dotenv(dotenv_path=dotenv_path, override=False)


def get_env_var(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def try_get_env_var(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value.strip() if isinstance(value, str) else None


