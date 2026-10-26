import json
import sys
import types as python_types
from types import SimpleNamespace

import pytest

from app.services.gemini import GeminiUnavailableError, generate_insights


def install_fake_sdk(monkeypatch, response_text=None, error=None, captured=None):
    class FakeModels:
        def generate_content(self, **kwargs):
            if captured is not None:
                captured.update(kwargs)
            if error:
                raise error
            return SimpleNamespace(text=response_text)

    class FakeClient:
        def __init__(self, **kwargs):
            if captured is not None:
                captured["client_options"] = kwargs
            self.models = FakeModels()

    genai = python_types.ModuleType("google.genai")
    genai.Client = FakeClient
    genai.__path__ = []
    sdk_types = python_types.ModuleType("google.genai.types")
    sdk_types.HttpOptions = lambda **kwargs: kwargs
    genai.types = sdk_types
    google = python_types.ModuleType("google")
    google.__path__ = []
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", sdk_types)


def sample_summary():
    return {
        "record_count": 4,
        "sensor_count": 1,
        "sensors": [
            {
                "name": "sensor_1",
                "minimum": 1,
                "maximum": 4,
                "mean": 2.5,
                "median": 2.5,
                "standard_deviation": 1.1,
            }
        ],
        "model_status": "No model predictions or evaluation metrics are available.",
    }


def test_returns_only_validated_structured_insights(monkeypatch):
    expected = {
        "summary": "Four values were analyzed.",
        "observations": "The values range from one to four.",
        "limitations": "No model prediction is available.",
    }
    captured = {}
    install_fake_sdk(monkeypatch, json.dumps(expected), captured=captured)

    result = generate_insights(
        sample_summary(), api_key="test-key", model="test-model"
    )

    assert result == expected
    assert captured["model"] == "test-model"
    assert "record_index" not in captured["contents"]
    assert "test-key" not in captured["contents"]
    assert captured["client_options"]["api_key"] == "test-key"


def test_missing_key_does_not_call_sdk():
    with pytest.raises(GeminiUnavailableError, match="not configured"):
        generate_insights(sample_summary(), api_key="", model="test-model")


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"summary":"ok","observations":"ok"}',
        '{"summary":"","observations":"ok","limitations":"ok"}',
        '{"summary":"' + ("x" * 2001) + '","observations":"ok","limitations":"ok"}',
    ],
)
def test_rejects_invalid_or_incomplete_responses(monkeypatch, response):
    install_fake_sdk(monkeypatch, response)

    with pytest.raises(GeminiUnavailableError):
        generate_insights(sample_summary(), api_key="test-key", model="test-model")


def test_api_failure_is_reported_without_provider_error_details(monkeypatch):
    install_fake_sdk(monkeypatch, error=RuntimeError("private transport detail"))

    with pytest.raises(GeminiUnavailableError, match="could not be reached") as error:
        generate_insights(sample_summary(), api_key="test-key", model="test-model")

    assert "private transport detail" not in str(error.value)
