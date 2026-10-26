import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    model_directory = Path(
        os.environ.get(
            "MODEL_ARTIFACT_DIRECTORY", str(PROJECT_ROOT / "models")
        ).strip()
    )
    if not model_directory.is_absolute():
        model_directory = PROJECT_ROOT / model_directory

    users: dict[str, str] = {}
    for number in range(1, 4):
        username = os.environ.get(f"AERIS_USER_{number}_USERNAME", "").strip()
        password_hash = os.environ.get(f"AERIS_USER_{number}_PASSWORD_HASH", "").strip()
        if bool(username) != bool(password_hash):
            raise RuntimeError(
                f"AERIS_USER_{number}_USERNAME and "
                f"AERIS_USER_{number}_PASSWORD_HASH must be set together."
            )
        if username and password_hash:
            if username in users:
                raise RuntimeError("Predefined usernames must be unique.")
            users[username] = password_hash

    return {
        "SECRET_KEY": os.environ.get("FLASK_SECRET_KEY"),
        "APP_ENV": os.environ.get("APP_ENV", "development").lower(),
        "AERIS_USERS": users,
        "NEON_DATABASE_URL": os.environ.get("NEON_DATABASE_URL", "").strip(),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", "").strip(),
        "GEMINI_MODEL": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        "MODEL_ARTIFACT_DIRECTORY": str(model_directory),
        "MAX_UPLOAD_BYTES": 5 * 1024 * 1024,
        "MAX_CONTENT_LENGTH": 5 * 1024 * 1024 + 64 * 1024,
        "MAX_TELEMETRY_RECORDS": 100_000,
        "MAX_TELEMETRY_SENSORS": 55,
        "MAX_CHART_POINTS": 400,
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "SESSION_COOKIE_SECURE": os.environ.get("APP_ENV", "").lower()
        == "production",
    }
