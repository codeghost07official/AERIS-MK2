import json
import logging
from typing import Any


logger = logging.getLogger(__name__)
MAX_RESPONSE_CHARS = 8_000


class GeminiUnavailableError(RuntimeError):
    """Raised when Gemini cannot return valid insights for a telemetry summary."""


def generate_insights(
    summary: dict[str, Any], *, api_key: str, model: str
) -> dict[str, str]:
    if not api_key:
        raise GeminiUnavailableError("Gemini API key is not configured.")
    if not model:
        raise GeminiUnavailableError("Gemini model name is not configured.")

    prompt = (
        "Explain this completed local telemetry analysis in plain language. "
        "Use only facts supported by the supplied JSON summary. Do not invent "
        "predictions, anomalies, causes, metrics, units, or certainty. If "
        "model_output is absent, explicitly say that no model prediction is "
        "available. If held_out_evaluation is present, explain that it is "
        "previously measured model-level performance, not proof about an "
        "individual upload prediction. Regression errors and classification "
        "metrics describe different tasks. Only the listed sensor summaries "
        "have per-sensor statistics; do not generalize to omitted channels. "
        "Sensor summaries contain only first/last values; model_output may "
        "contain only a bounded prediction sample, not a full sequence. Do not "
        "claim a sustained temporal trend. Distinguish a model-flagged record "
        "from a confirmed anomaly. Return only a JSON object with "
        "exactly three string fields: summary, observations, limitations. "
        "Each field must be concise, and limitations must recommend human "
        "review where appropriate."
        "\n\nAnalysis summary:\n"
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        logger.warning("Gemini SDK is not installed.")
        raise GeminiUnavailableError("Gemini support is not installed.") from exc

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=15_000),
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        response_text = getattr(response, "text", None)
    except Exception as exc:
        # SDK transport and API exceptions vary; avoid logging exception text,
        # which could contain request details or sensitive configuration.
        logger.warning("Gemini request failed (%s).", type(exc).__name__)
        raise GeminiUnavailableError(
            "Gemini could not be reached. Local analysis is still available."
        ) from exc

    if not isinstance(response_text, str) or len(response_text) > MAX_RESPONSE_CHARS:
        raise GeminiUnavailableError("Gemini returned an empty or oversized response.")
    try:
        parsed = json.loads(response_text)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise GeminiUnavailableError("Gemini returned an invalid response format.") from exc

    required_fields = ("summary", "observations", "limitations")
    if (
        not isinstance(parsed, dict)
        or set(parsed) != set(required_fields)
        or any(
            not isinstance(parsed[field], str)
            or not parsed[field].strip()
            or len(parsed[field]) > 2_000
            for field in required_fields
        )
    ):
        raise GeminiUnavailableError("Gemini returned an incomplete response.")
    return {field: parsed[field].strip() for field in required_fields}
