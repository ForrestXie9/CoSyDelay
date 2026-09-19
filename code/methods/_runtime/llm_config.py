# -*- coding: utf-8 -*-
"""Configuration for the OpenAI-compatible LLM gateway.

All runtime settings can be overridden through environment variables.  The
original CoSyDelay request leaves temperature, top-p, and the output-token cap
unset; ``None`` below preserves that provider-default behavior and is recorded
explicitly in audits.  API keys never have a source-code default.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Mapping, Optional


JIEKOU_API_ENDPOINT = "llm-api.net"
# Credentials must never have a source-code fallback.  Configure
# ``LLM_API_KEY`` in the process environment or a secret manager.
JIEKOU_API_KEY = ""
JIEKOU_MODEL = "gpt-4.1-mini"
JIEKOU_REQUEST_PATH = "/v1/chat/completions"

DEFAULT_TEMPERATURE = None
DEFAULT_TOP_P = None
DEFAULT_MAX_TOKENS = None
DEFAULT_SEED = 20260712
DEFAULT_TIMEOUT = 30.0

# Paper-facing competitor labels may differ from gateway model ids on llm-api.net.
GATEWAY_MODEL_ALIASES = {
    "gpt-oss-20b": "gpt-oss-120b",
    "grok-code-fast-1": "grok-build-0.1",
}



def resolve_gateway_model(logical_model: str) -> str:
    """Map a manuscript/backend label to the provider model id."""
    model = logical_model.strip()
    if not model:
        raise ValueError("logical_model must be non-empty")
    return GATEWAY_MODEL_ALIASES.get(model, model)


@dataclass(frozen=True)
class LLMConfig:
    endpoint: str
    api_key: str
    model: str
    request_path: str
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]
    seed: int
    timeout: float
    audit_log: Optional[str]
    audit_include_content: bool

    @property
    def sampling_parameters(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }

    @property
    def sampling_parameter_sources(self) -> dict:
        return {
            "temperature": (
                "explicit_request" if self.temperature is not None
                else "provider_default_not_sent"
            ),
            "top_p": (
                "explicit_request" if self.top_p is not None
                else "provider_default_not_sent"
            ),
            "max_tokens": (
                "explicit_request" if self.max_tokens is not None
                else "provider_default_not_sent"
            ),
            "seed": "explicit_request",
        }


def _parse_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: Optional[float] = None,
    minimum_inclusive: bool = True,
) -> float:
    raw = environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {raw!r}")

    below_minimum = value < minimum if minimum_inclusive else value <= minimum
    if below_minimum or (maximum is not None and value > maximum):
        lower = "[" if minimum_inclusive else "("
        upper = f", {maximum}]" if maximum is not None else ", infinity)"
        raise ValueError(f"{name} must be in {lower}{minimum}{upper}, got {value}")
    return value


def _parse_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    raw = environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _parse_bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    raw = environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of true/false, 1/0, yes/no, or on/off; got {raw!r}"
    )


def _parse_optional_float(
    environ: Mapping[str, str],
    name: str,
    *,
    minimum: float,
    maximum: Optional[float] = None,
    minimum_inclusive: bool = True,
) -> Optional[float]:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return None
    return _parse_float(
        environ,
        name,
        0.0,
        minimum=minimum,
        maximum=maximum,
        minimum_inclusive=minimum_inclusive,
    )


def _parse_optional_int(
    environ: Mapping[str, str], name: str, *, minimum: int
) -> Optional[int]:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return None
    return _parse_int(environ, name, minimum, minimum=minimum)


def load_llm_config(environ: Optional[Mapping[str, str]] = None) -> LLMConfig:
    """Parse and validate one immutable runtime configuration."""
    values = os.environ if environ is None else environ
    endpoint = values.get("LLM_API_ENDPOINT", JIEKOU_API_ENDPOINT).strip()
    api_key = values.get("LLM_API_KEY", JIEKOU_API_KEY).strip()
    model = values.get("LLM_MODEL", JIEKOU_MODEL).strip()
    request_path = values.get("LLM_REQUEST_PATH", JIEKOU_REQUEST_PATH).strip()
    audit_log = values.get("LLM_AUDIT_LOG", "").strip() or None

    if not endpoint:
        raise ValueError("LLM_API_ENDPOINT is empty")
    if not api_key:
        raise ValueError("LLM_API_KEY is empty; set it in the process environment")
    if any(ord(character) < 32 or ord(character) == 127 for character in api_key):
        raise ValueError(
            "LLM_API_KEY contains a control character; reset it from the "
            "provider token value without copied line breaks"
        )
    if not model:
        raise ValueError("LLM_MODEL is empty")
    if not request_path:
        raise ValueError("LLM_REQUEST_PATH is empty")
    if not request_path.startswith("/"):
        raise ValueError("LLM_REQUEST_PATH must start with '/'")

    return LLMConfig(
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        request_path=request_path,
        temperature=_parse_optional_float(
            values,
            "LLM_TEMPERATURE",
            minimum=0.0,
            maximum=2.0,
        ),
        top_p=_parse_optional_float(
            values,
            "LLM_TOP_P",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        ),
        max_tokens=_parse_optional_int(values, "LLM_MAX_TOKENS", minimum=1),
        seed=_parse_int(values, "LLM_SEED", DEFAULT_SEED, minimum=0),
        timeout=_parse_float(
            values,
            "LLM_TIMEOUT",
            DEFAULT_TIMEOUT,
            minimum=0.0,
            minimum_inclusive=False,
        ),
        audit_log=audit_log,
        audit_include_content=_parse_bool(
            values, "LLM_AUDIT_INCLUDE_CONTENT", False
        ),
    )
