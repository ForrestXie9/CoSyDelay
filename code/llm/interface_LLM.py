"""Compatibility wrapper around the OpenAI-compatible API client."""
from llm.api_general import InterfaceAPI


class InterfaceLLM:
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
        # The caller supplies the runtime credential from ``LLM_API_KEY``.
        # Do not replace it with a source-controlled value: doing so makes
        # key rotation ineffective and can send requests to the provider with
        # a stale credential.
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.request_path = request_path

        self.interface_llm = InterfaceAPI(
            self.api_endpoint,
            self.api_key,
            self.model_LLM,
            self.debug_mode,
            self.request_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            timeout=timeout,
        )

    def get_response_with_metadata(self, prompt_content):
        """Return the provider content and non-secret response metadata."""
        return self.interface_llm.get_response(prompt_content)

    def get_response(self, prompt_content):
        """Preserve the historical direct-call string interface."""
        return self.get_response_with_metadata(prompt_content)["content"]
