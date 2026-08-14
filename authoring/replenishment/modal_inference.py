"""Llama inference served from a Modal endpoint, behind a Modal proxy auth token.

A sibling of worker.py's LlamaBatchModel: the same BatchModel protocol
(model_id, model_revision, generate), reached over HTTP instead of loaded
in-process. Swapping providers is choosing which of the two adapters
authoring.replenishment.cli builds for the worker -- nothing else in the
pipeline (generate_batch, the job repository, the stage dispatch in
worker.py) knows or needs to know which one is in use.

Any failure to reach the endpoint or get a well-formed completion back is a
ModelUnavailableError, exactly like the local adapter -- that is what routes
a replenishment job to waiting_for_model instead of failing it.
"""

import os
from collections.abc import Mapping

import httpx

from authoring.grounded_batch import GenerationOutcome
from authoring.replenishment.worker import ModelUnavailableError

ENDPOINT_VARIABLE = "MODAL_INFERENCE_ENDPOINT"
KEY_VARIABLE = "MODAL_PROXY_TOKEN_ID"
SECRET_VARIABLE = "MODAL_PROXY_TOKEN_SECRET"
# Same identity variables the local adapter reads, so provenance (model_id,
# model_revision) is consistent regardless of where inference actually runs.
MODEL_ID_VARIABLE = "MODEL_REPOSITORY"
MODEL_REVISION_VARIABLE = "MODEL_REVISION"

# Generous: a cold-started GPU container can take tens of seconds to spin up
# before it ever starts generating. A timeout this short would misreport a
# slow-but-working cold start as an outage.
DEFAULT_TIMEOUT = 180.0


class ModalBatchModel:
    def __init__(
        self,
        *,
        endpoint: str | None = None,
        key: str | None = None,
        secret: str | None = None,
        model_revision: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        environ = os.environ if environ is None else environ
        self._endpoint = endpoint if endpoint is not None else environ.get(ENDPOINT_VARIABLE, "")
        self._key = key if key is not None else environ.get(KEY_VARIABLE, "")
        self._secret = secret if secret is not None else environ.get(SECRET_VARIABLE, "")
        self._model_id = environ.get(MODEL_ID_VARIABLE, "")
        # An explicit pin (constructor arg or MODEL_REVISION env var) is deliberate
        # operator intent and is never silently overridden. Absent a pin, _request()
        # opportunistically records the exact commit hash the server resolved for the
        # loaded model (a real, pinned identifier) the first time a response actually
        # includes one -- see _request() below. Older-deployed endpoints that predate
        # this field simply never send it, and "unknown" stands.
        self._pinned_model_revision = model_revision or environ.get(MODEL_REVISION_VARIABLE) or None
        self._model_revision = self._pinned_model_revision or "unknown"
        self.client = client or httpx.Client()
        self.timeout = timeout

    @property
    def model_id(self) -> str:
        if not self._model_id.strip():
            raise ModelUnavailableError(f"{MODEL_ID_VARIABLE} is not set.")
        return self._model_id

    @property
    def model_revision(self) -> str:
        return self._model_revision

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._key.strip() and self._secret.strip():
            headers["Modal-Key"] = self._key
            headers["Modal-Secret"] = self._secret
        return headers

    def generate(
        self, messages: list[dict[str, str]], seed: int, generation_parameters: dict
    ) -> str:
        return self._request(messages, seed, generation_parameters).text

    def generate_with_metadata(
        self, messages: list[dict[str, str]], seed: int, generation_parameters: dict
    ) -> GenerationOutcome:
        return self._request(messages, seed, generation_parameters)

    def _request(
        self, messages: list[dict[str, str]], seed: int, generation_parameters: dict
    ) -> GenerationOutcome:
        if not self._endpoint.strip():
            raise ModelUnavailableError(f"{ENDPOINT_VARIABLE} is not set.")

        try:
            response = self.client.post(
                self._endpoint,
                json={
                    "messages": messages,
                    "seed": seed,
                    "generation_parameters": generation_parameters,
                },
                headers=self._headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(f"Modal endpoint request failed: {exc}") from exc
        except ValueError as exc:
            raise ModelUnavailableError(f"Modal endpoint returned invalid JSON: {exc}") from exc

        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ModelUnavailableError(
                "Modal endpoint response is missing a non-empty 'text' field."
            )
        if self._pinned_model_revision is None:
            server_revision = payload.get("model_revision") if isinstance(payload, dict) else None
            if isinstance(server_revision, str) and server_revision.strip():
                self._model_revision = server_revision
        return GenerationOutcome(
            text=text,
            finish_reason=payload.get("finish_reason", "unknown"),
            input_tokens=payload.get("input_tokens", 0),
            output_tokens=payload.get("output_tokens", 0),
            max_new_tokens=payload.get("max_new_tokens", generation_parameters.get("max_new_tokens", 800)),
        )
