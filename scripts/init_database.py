import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    uploaded_filename TEXT NOT NULL,
    channel_id TEXT,
    record_count INTEGER NOT NULL CHECK (record_count >= 0),
    anomaly_count INTEGER CHECK (anomaly_count IS NULL OR anomaly_count >= 0),
    prediction_summary TEXT NOT NULL,
    evaluation_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    descriptive_statistics JSONB NOT NULL DEFAULT '{}'::jsonb,
    gemini_insights TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

MIGRATION = """
ALTER TABLE analyses
ADD COLUMN IF NOT EXISTS descriptive_statistics JSONB NOT NULL DEFAULT '{}'::jsonb;
"""


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")
    database_url = os.environ.get("NEON_DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit("Set NEON_DATABASE_URL before initializing the database.")

    try:
        with psycopg.connect(database_url, connect_timeout=10) as connection:
            connection.execute(SCHEMA)
            connection.execute(MIGRATION)
    except psycopg.Error as exc:
        raise SystemExit(
            f"Database initialization failed ({type(exc).__name__}). "
            "Check the Neon connection settings."
        ) from None
    print("Database schema is ready.")


if __name__ == "__main__":
    main()
