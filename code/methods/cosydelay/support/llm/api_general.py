"""Minimal OpenAI-compatible chat-completions client."""
from __future__ import annotations

import http.client
import json
import time
from typing import Any, Dict, Optional


class LLMAPIError(RuntimeError):
    """Provider/transport error carrying non-secret request metadata."""

    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.metadata = metadata or {}


class InterfaceAPI:
    def __init__(
        self,
        api_endpoint,
        api_key,
        model_LLM,
        debug_mode,
        request_path="/v1/chat/completions",
        *,
        temperature=None,
        top_p=None,
        max_tokens=None,
        seed=None,
        timeout=30.0,
    ):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.request_path = request_path
        self.temperature = (
            None if temperature is None else float(temperature)
        )
        self.top_p = None if top_p is None else float(top_p)
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        # Gemini/Google generation_config.seed must fit signed int32.
        self.seed = (
            None if seed is None else int(seed) & 0x7FFFFFFF
        )
        self.timeout = float(timeout)

    def _metadata(self, http_status, started, **extra):
        metadata = {
            "http_status": http_status,
            "latency_seconds": float(time.perf_counter() - started),
            "usage": None,
        }
        metadata.update(extra)
        return metadata

    def _provider_error_detail(self, data):
        """Return a bounded, non-secret provider error description.

        OpenAI-compatible providers commonly put the actionable reason for a
        4xx response in the JSON body (for example an unsupported sampling
        parameter).  Keeping that reason in the audit trail makes requests
        diagnosable without recording the request headers or API key.
        """
        try:
            text = data.decode("utf-8", errors="replace").strip()
        except Exception:
            text = "<unreadable provider error body>"
        if self.api_key:
            text = text.replace(self.api_key, "[REDACTED]")
        return text[:1000] or "<empty provider error body>"

    def get_response(self, prompt_content):
        """Return ``{"content": str, "metadata": dict}``.

        The metadata intentionally excludes request headers and the API key.
        """
        payload = {
            "model": self.model_LLM,
            "messages": [{"role": "user", "content": prompt_content}],
        }
        optional_sampling = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }
        payload.update(
            {
                name: value
                for name, value in optional_sampling.items()
                if value is not None
            }
        )
        payload_explanation = json.dumps(payload, ensure_ascii=False)

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "CoSyDelay/1.0",
            "Content-Type": "application/json",
            "x-api2d-no-cache": "1",
        }

        conn = None
        response = None
        started = time.perf_counter()
        try:
            conn = http.client.HTTPSConnection(
                self.api_endpoint, timeout=self.timeout
            )
            # ``http.client`` encodes string bodies as Latin-1.  Prompts and
            # provider error context can contain non-ASCII text, so send the
            # JSON payload as explicit UTF-8 bytes.
            conn.request(
                "POST",
                self.request_path,
                payload_explanation.encode("utf-8"),
                headers,
            )
            response = conn.getresponse()
            data = response.read()

            if response.status < 200 or response.status >= 300:
                provider_error = self._provider_error_detail(data)
                raise LLMAPIError(
                    f"LLM API returned HTTP {response.status}: {provider_error}",
                    self._metadata(
                        response.status,
                        started,
                        provider_error=provider_error,
                    ),
                )

            try:
                json_data = json.loads(data)
            except (TypeError, ValueError) as exc:
                raise LLMAPIError(
                    "LLM API returned invalid JSON",
                    self._metadata(response.status, started),
                ) from exc

            choices = json_data.get("choices", [])
            if not choices:
                raise LLMAPIError(
                    "LLM API returned no choices",
                    self._metadata(
                        response.status,
                        started,
                        usage=json_data.get("usage"),
                        response_id=json_data.get("id"),
                    ),
                )

            content = choices[0].get("message", {}).get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise LLMAPIError(
                    "LLM API returned empty content",
                    self._metadata(
                        response.status,
                        started,
                        usage=json_data.get("usage"),
                        response_id=json_data.get("id"),
                    ),
                )

            return {
                "content": content,
                "metadata": self._metadata(
                    response.status,
                    started,
                    usage=json_data.get("usage"),
                    response_id=json_data.get("id"),
                    response_model=json_data.get("model"),
                    finish_reason=choices[0].get("finish_reason"),
                ),
            }
        except LLMAPIError:
            raise
        except Exception as exc:
            http_status = response.status if response is not None else None
            metadata = self._metadata(
                http_status,
                started,
                transport_error_type=type(exc).__name__,
            )
            if self.debug_mode:
                print(f"Error in API request: {type(exc).__name__}: {exc}")
            raise LLMAPIError(
                f"LLM API request failed ({type(exc).__name__}): {exc}",
                metadata,
            ) from exc
        finally:
            if conn is not None:
                conn.close()
