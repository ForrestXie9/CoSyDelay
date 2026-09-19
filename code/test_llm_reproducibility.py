import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm.api_general import InterfaceAPI, LLMAPIError
from llm_config import load_llm_config
from llm_integration import run_llm, run_llm_with_metadata


BASE_ENV = {
    "LLM_API_ENDPOINT": "example.invalid",
    "LLM_API_KEY": "unit-test-secret-key",
    "LLM_MODEL": "test-model-snapshot",
    "LLM_REQUEST_PATH": "/v1/chat/completions",
    "LLM_TEMPERATURE": "0.25",
    "LLM_TOP_P": "0.8",
    "LLM_MAX_TOKENS": "321",
    "LLM_SEED": "17",
    "LLM_TIMEOUT": "12.5",
}


class _FakeHTTPResponse:
    status = 200

    def read(self):
        return json.dumps({
            "id": "response-1",
            "model": "provider-model-version",
            "choices": [{
                "message": {"content": "  generated expression  "},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
            },
        }).encode("utf-8")


class _FakeHTTPSConnection:
    instances = []

    def __init__(self, endpoint, timeout):
        self.endpoint = endpoint
        self.timeout = timeout
        self.request_args = None
        self.closed = False
        self.__class__.instances.append(self)

    def request(self, method, path, body, headers):
        self.request_args = (method, path, body, headers)

    def getresponse(self):
        return _FakeHTTPResponse()

    def close(self):
        self.closed = True


class LLMConfigurationTests(unittest.TestCase):
    def test_original_defaults_leave_three_sampling_fields_unset(self):
        environment = {
            key: value
            for key, value in BASE_ENV.items()
            if key not in {"LLM_TEMPERATURE", "LLM_TOP_P", "LLM_MAX_TOKENS"}
        }
        config = load_llm_config(environment)
        self.assertIsNone(config.temperature)
        self.assertIsNone(config.top_p)
        self.assertIsNone(config.max_tokens)
        self.assertEqual(config.seed, 17)
        self.assertEqual(
            config.sampling_parameter_sources,
            {
                "temperature": "provider_default_not_sent",
                "top_p": "provider_default_not_sent",
                "max_tokens": "provider_default_not_sent",
                "seed": "explicit_request",
            },
        )

    def test_explicit_environment_parsing(self):
        config = load_llm_config(BASE_ENV)
        self.assertEqual(config.endpoint, "example.invalid")
        self.assertEqual(config.model, "test-model-snapshot")
        self.assertEqual(config.temperature, 0.25)
        self.assertEqual(config.top_p, 0.8)
        self.assertEqual(config.max_tokens, 321)
        self.assertEqual(config.seed, 17)
        self.assertEqual(config.timeout, 12.5)
        self.assertFalse(config.audit_include_content)

    def test_blank_key_and_invalid_sampling_values_fail_fast(self):
        # This private experiment keeps an embedded provider key as its default.
        # An explicit empty environment override must still fail clearly.
        missing_key = dict(BASE_ENV, LLM_API_KEY="")
        with self.assertRaisesRegex(ValueError, "LLM_API_KEY is empty"):
            load_llm_config(missing_key)

        bad_top_p = dict(BASE_ENV, LLM_TOP_P="0")
        with self.assertRaisesRegex(ValueError, "LLM_TOP_P must be in"):
            load_llm_config(bad_top_p)

        bad_timeout = dict(BASE_ENV, LLM_TIMEOUT="not-a-number")
        with self.assertRaisesRegex(ValueError, "LLM_TIMEOUT must be a number"):
            load_llm_config(bad_timeout)

        non_finite_temperature = dict(BASE_ENV, LLM_TEMPERATURE="nan")
        with self.assertRaisesRegex(ValueError, "LLM_TEMPERATURE must be finite"):
            load_llm_config(non_finite_temperature)

    @patch("llm.api_general.http.client.HTTPSConnection", _FakeHTTPSConnection)
    def test_api_sends_sampling_parameters_and_returns_metadata(self):
        _FakeHTTPSConnection.instances.clear()
        client = InterfaceAPI(
            "example.invalid",
            "unit-test-secret-key",
            "test-model-snapshot",
            False,
            temperature=0.25,
            top_p=0.8,
            max_tokens=321,
            seed=17,
            timeout=12.5,
        )
        result = client.get_response("中文 prompt")
        connection = _FakeHTTPSConnection.instances[-1]
        request_body = connection.request_args[2]
        self.assertIsInstance(request_body, bytes)
        payload = json.loads(request_body.decode("utf-8"))

        self.assertEqual(connection.timeout, 12.5)
        self.assertEqual(payload["temperature"], 0.25)
        self.assertEqual(payload["top_p"], 0.8)
        self.assertEqual(payload["max_tokens"], 321)
        self.assertEqual(payload["seed"], 17)
        self.assertEqual(payload["messages"][0]["content"], "中文 prompt")
        self.assertEqual(result["content"], "  generated expression  ")
        self.assertEqual(result["metadata"]["http_status"], 200)
        self.assertEqual(result["metadata"]["usage"]["total_tokens"], 16)
        self.assertEqual(result["metadata"]["finish_reason"], "stop")
        self.assertTrue(connection.closed)

    @patch("llm.api_general.http.client.HTTPSConnection", _FakeHTTPSConnection)
    def test_api_omits_unset_sampling_parameters(self):
        _FakeHTTPSConnection.instances.clear()
        client = InterfaceAPI(
            "example.invalid",
            "unit-test-secret-key",
            "test-model-snapshot",
            False,
            seed=17,
            timeout=12.5,
        )
        client.get_response("provider defaults")
        request_body = _FakeHTTPSConnection.instances[-1].request_args[2]
        payload = json.loads(request_body.decode("utf-8"))
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(payload["seed"], 17)


class LLMAuditTests(unittest.TestCase):
    def _environment(self, audit_path, include_content=False):
        return {
            **BASE_ENV,
            "LLM_AUDIT_LOG": str(audit_path),
            "LLM_AUDIT_INCLUDE_CONTENT": "true" if include_content else "false",
        }

    def test_retry_audit_is_complete_and_run_llm_stays_string_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            audit_path = Path(directory) / "audit.jsonl"
            first_error = LLMAPIError(
                "provider rejected unit-test-secret-key",
                {
                    "http_status": 429,
                    "latency_seconds": 0.15,
                    "usage": None,
                },
            )
            success = {
                "content": "  final answer  ",
                "metadata": {
                    "http_status": 200,
                    "latency_seconds": 0.25,
                    "usage": {"total_tokens": 22},
                    "response_id": "response-2",
                    "response_model": "provider-model-version",
                    "finish_reason": "stop",
                },
            }

            with patch.dict(
                os.environ, self._environment(audit_path), clear=True
            ), patch("llm_integration.InterfaceLLM") as client_class:
                client_class.return_value.get_response_with_metadata.side_effect = [
                    first_error,
                    success,
                ]
                content = run_llm("audit prompt", max_retries=2, retry_delay=0)

            self.assertEqual(content, "final answer")
            constructor_kwargs = client_class.call_args.kwargs
            self.assertEqual(constructor_kwargs["temperature"], 0.25)
            self.assertEqual(constructor_kwargs["top_p"], 0.8)
            self.assertEqual(constructor_kwargs["max_tokens"], 321)
            self.assertEqual(constructor_kwargs["seed"], 17)
            self.assertEqual(constructor_kwargs["timeout"], 12.5)

            raw_audit = audit_path.read_text(encoding="utf-8")
            self.assertNotIn("unit-test-secret-key", raw_audit)
            records = [json.loads(line) for line in raw_audit.splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["status"], "error")
            self.assertEqual(records[0]["http_status"], 429)
            self.assertEqual(records[0]["attempt"], 1)
            self.assertTrue(records[0]["will_retry"])
            self.assertEqual(records[1]["status"], "success")
            self.assertEqual(records[1]["attempt"], 2)
            self.assertTrue(records[1]["retry"])
            self.assertEqual(records[1]["usage"]["total_tokens"], 22)
            self.assertNotIn("prompt", records[1])
            self.assertNotIn("response", records[1])
            expected_hash = hashlib.sha256(b"audit prompt").hexdigest()
            self.assertEqual(records[1]["prompt_sha256"], expected_hash)

    def test_optional_content_is_logged_but_api_key_is_always_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            audit_path = Path(directory) / "audit.jsonl"
            result = {
                "content": "response mentions unit-test-secret-key",
                "metadata": {
                    "http_status": 200,
                    "latency_seconds": 0.01,
                    "usage": {},
                },
            }
            prompt = "prompt mentions unit-test-secret-key"
            with patch.dict(
                os.environ,
                self._environment(audit_path, include_content=True),
                clear=True,
            ), patch("llm_integration.InterfaceLLM") as client_class:
                client_class.return_value.get_response_with_metadata.return_value = result
                envelope = run_llm_with_metadata(prompt, max_retries=1)

            self.assertEqual(
                envelope["content"], "response mentions unit-test-secret-key"
            )
            raw_audit = audit_path.read_text(encoding="utf-8")
            self.assertNotIn("unit-test-secret-key", raw_audit)
            record = json.loads(raw_audit)
            self.assertEqual(record["prompt"], "prompt mentions [REDACTED]")
            self.assertEqual(record["response"], "response mentions [REDACTED]")


if __name__ == "__main__":
    unittest.main()
