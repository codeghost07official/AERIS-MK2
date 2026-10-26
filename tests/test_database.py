import json
from types import SimpleNamespace

from app.services import database


class FakeConnection:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, params=None):
        self.calls.append((query, params))
        return SimpleNamespace(fetchall=lambda: [])


def test_history_query_is_limited_and_parameterized(monkeypatch):
    connection = FakeConnection()
    monkeypatch.setattr(
        database.psycopg,
        "connect",
        lambda *args, **kwargs: connection,
    )

    result = database.get_recent_analyses("postgresql://example", limit=10)

    assert result == []
    query, params = connection.calls[0]
    assert "LIMIT %s" in query
    assert params == (10,)


def test_save_analysis_persists_only_summary_and_insights(monkeypatch):
    connection = FakeConnection()
    monkeypatch.setattr(
        database.psycopg,
        "connect",
        lambda *args, **kwargs: connection,
    )
    insights = {
        "summary": "Summary",
        "observations": "Observation",
        "limitations": "Limitations",
    }

    database.save_analysis(
        "postgresql://example",
        filename="telemetry.csv",
        record_count=250,
        sensor_count=3,
        sensor_summaries=[
            {
                "name": "sensor_1",
                "minimum": 1,
                "maximum": 3,
                "mean": 2,
                "median": 2,
                "standard_deviation": 1,
                "first_value": 1,
                "last_value": 3,
                "net_change": 2,
                "histogram": {"labels": ["1-3"], "counts": [3]},
            }
        ],
        gemini_insights=insights,
        model_result=None,
    )

    query, params = connection.calls[0]
    assert "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)" in query
    assert params[0:3] == ("telemetry.csv", None, 250)
    assert params[3] is None and "unavailable" in params[4]
    metrics = json.loads(params[5])
    assert metrics == {}
    statistics = json.loads(params[6])
    assert len(statistics) == 1
    assert "histogram" not in statistics[0]
    assert params[7].find('"observations"') >= 0


def test_save_analysis_allows_gemini_to_be_unavailable(monkeypatch):
    connection = FakeConnection()
    monkeypatch.setattr(
        database.psycopg,
        "connect",
        lambda *args, **kwargs: connection,
    )

    database.save_analysis(
        "postgresql://example",
        filename="telemetry.csv",
        record_count=1,
        sensor_count=1,
        sensor_summaries=[],
        gemini_insights=None,
        model_result=None,
    )

    assert connection.calls[0][1][7] is None
