"""Reproducible, audited access to the configured LLM provider."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Tuple

from llm.api_general import LLMAPIError
from llm.interface_LLM import InterfaceLLM
from llm_config import LLMConfig, load_llm_config, resolve_gateway_model


_AUDIT_LOCK = threading.Lock()


def _fs_path(path: Path | str) -> Path:
    text = os.path.abspath(str(path))
    if os.name != "nt" or text.startswith("\\\\?\\"):
        return Path(text)
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text[2:])
    return Path("\\\\?\\" + text)


def _resolve_llm_config() -> Tuple[str, str, str, str]:
    """Retain the former private helper's tuple contract."""
    config = load_llm_config()
    return config.endpoint, config.api_key, config.model, config.request_path


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _redact_secret(value: Any, secret: str) -> Any:
    """Remove an API key even if it was accidentally included in content/errors."""
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, dict):
        return {key: _redact_secret(item, secret) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_secret(item, secret) for item in value]
    return value


def _write_audit(config: LLMConfig, record: Dict[str, Any]) -> None:
    if not config.audit_log:
        return
    path = _fs_path(Path(config.audit_log).expanduser())
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _redact_secret(record, config.api_key)
    line = json.dumps(sanitized, ensure_ascii=False, sort_keys=True) + "\n"
    with _AUDIT_LOCK:
        with path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(line)
            handle.flush()


def _base_audit_record(
    config: LLMConfig,
    prompt: str,
    attempt: int,
    max_retries: int,
) -> Dict[str, Any]:
    record = {
        "utc": _utc_now(),
        "model": config.model,
        "sampling_parameters": config.sampling_parameters,
        "sampling_parameter_sources": config.sampling_parameter_sources,
        "timeout_seconds": config.timeout,
        "prompt_sha256": _sha256_text(prompt),
        "attempt": attempt,
        "max_attempts": max_retries,
        "retry": attempt > 1,
        "retry_index": attempt - 1,
        "will_retry": False,
        "http_status": None,
        "status": None,
        "latency_seconds": None,
        "usage": None,
        "error": None,
    }
    if config.audit_include_content:
        record["prompt"] = prompt
    return record


def run_llm_with_metadata(
    prompt: str,
    max_retries: int = 5,
    retry_delay: float = 2.0,
) -> Dict[str, Any]:
    """Return generated content plus provider metadata, auditing every attempt."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    if retry_delay < 0:
        raise ValueError("retry_delay must be >= 0")

    config = load_llm_config()
    client = InterfaceLLM(
        config.endpoint,
        config.api_key,
        resolve_gateway_model(config.model),
        False,
        request_path=config.request_path,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        seed=config.seed,
        timeout=config.timeout,
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        audit = _base_audit_record(config, prompt, attempt, max_retries)
        attempt_started = time.perf_counter()
        try:
            result = client.get_response_with_metadata(prompt)
            if not isinstance(result, dict):
                raise RuntimeError("LLM client returned an invalid response envelope")
            content = result.get("content")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("Empty response from LLM")
            content = content.strip()
            provider_metadata = result.get("metadata") or {}
            if not isinstance(provider_metadata, dict):
                raise RuntimeError("LLM client returned invalid response metadata")
        except Exception as exc:
            last_error = exc
            provider_metadata = (
                exc.metadata
                if isinstance(exc, LLMAPIError)
                else {}
            )
            audit.update({
                "http_status": provider_metadata.get("http_status"),
                "status": "error",
                "latency_seconds": provider_metadata.get(
                    "latency_seconds", time.perf_counter() - attempt_started
                ),
                "usage": provider_metadata.get("usage"),
                # api_general.py has already bounded and redacted this field.
                # It is intentionally only written for failed provider calls.
                "provider_error": provider_metadata.get("provider_error"),
                "will_retry": attempt < max_retries,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            })
            _write_audit(config, audit)
            if attempt < max_retries and retry_delay:
                time.sleep(retry_delay)
            continue

        audit.update({
            "http_status": provider_metadata.get("http_status"),
            "status": "success",
            "latency_seconds": provider_metadata.get(
                "latency_seconds", time.perf_counter() - attempt_started
            ),
            "usage": provider_metadata.get("usage"),
            "response_sha256": _sha256_text(content),
            "response_id": provider_metadata.get("response_id"),
            "response_model": provider_metadata.get("response_model"),
            "finish_reason": provider_metadata.get("finish_reason"),
        })
        if config.audit_include_content:
            audit["response"] = content
        _write_audit(config, audit)

        metadata = dict(provider_metadata)
        metadata.update({
            "attempt": attempt,
            "retry_index": attempt - 1,
            "model": config.model,
            "sampling_parameters": config.sampling_parameters,
            "sampling_parameter_sources": config.sampling_parameter_sources,
            "prompt_sha256": audit["prompt_sha256"],
            "response_sha256": audit["response_sha256"],
        })
        return {"content": content, "metadata": metadata}

    safe_error = _redact_secret(str(last_error), config.api_key)
    raise RuntimeError(
        f"Failed to generate valid LLM output after {max_retries} attempts: "
        f"{safe_error}"
    ) from last_error


def run_llm(prompt: str, max_retries: int = 5, retry_delay: float = 2.0) -> str:
    """Historical string-returning interface used by the expression pipeline."""
    return run_llm_with_metadata(prompt, max_retries, retry_delay)["content"]
