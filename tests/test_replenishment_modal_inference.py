"""ModalBatchModel driven by a mock transport -- nothing here opens a socket.

Mirrors tests/test_retrieval_http.py's pattern for the two other adapters
that touch the network.
"""

import json

import httpx
import pytest

from authoring.replenishment.modal_inference import (
    ENDPOINT_VARIABLE,
    KEY_VARIABLE,
    MODEL_ID_VARIABLE,
    SECRET_VARIABLE,
    ModalBatchModel,
)
from authoring.replenishment.worker import ModelUnavailableError

ENDPOINT = "https://workspace--llama-quiz-generate.modal.run"

ENV = {
    ENDPOINT_VARIABLE: ENDPOINT,
    KEY_VARIABLE: "wk-test-key",
    SECRET_VARIABLE: "ws-test-secret",
    MODEL_ID_VARIABLE: "meta-llama/Llama-3.1-8B-Instruct",
}


def client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def model(environ=ENV, client=None) -> ModalBatchModel:
    return ModalBatchModel(client=client, environ=environ)


def test_model_id_and_revision_read_from_environment():
    instance = model()
    assert instance.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert instance.model_revision == "unknown"


def test_model_id_missing_raises_model_unavailable():
    instance = model(environ={**ENV, MODEL_ID_VARIABLE: ""})
    with pytest.raises(ModelUnavailableError, match="MODEL_REPOSITORY"):
        _ = instance.model_id


def test_generate_posts_messages_seed_and_parameters_with_proxy_headers():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"text": "generated completion"})

    instance = model(client=client_for(handler))
    result = instance.generate([{"role": "user", "content": "hi"}], 42, {"max_new_tokens": 10})

    assert result == "generated completion"
    assert captured["url"] == ENDPOINT
    assert captured["headers"]["modal-key"] == "wk-test-key"
    assert captured["headers"]["modal-secret"] == "ws-test-secret"
    assert captured["body"] == {
        "messages": [{"role": "user", "content": "hi"}],
        "seed": 42,
        "generation_parameters": {"max_new_tokens": 10},
    }


def test_generate_without_proxy_credentials_omits_the_headers():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"text": "ok"})

    instance = model(
        environ={ENDPOINT_VARIABLE: ENDPOINT, MODEL_ID_VARIABLE: "m"},
        client=client_for(handler),
    )
    instance.generate([], 1, {})

    assert "modal-key" not in captured["headers"]
    assert "modal-secret" not in captured["headers"]


def test_missing_endpoint_raises_model_unavailable():
    instance = model(environ={**ENV, ENDPOINT_VARIABLE: ""})
    with pytest.raises(ModelUnavailableError, match="MODAL_INFERENCE_ENDPOINT"):
        instance.generate([], 1, {})


@pytest.mark.parametrize("status_code", [401, 429, 500, 503])
def test_non_2xx_status_raises_model_unavailable(status_code):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text="error")

    instance = model(client=client_for(handler))
    with pytest.raises(ModelUnavailableError):
        instance.generate([], 1, {})


def test_connection_failure_raises_model_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    instance = model(client=client_for(handler))
    with pytest.raises(ModelUnavailableError, match="request failed"):
        instance.generate([], 1, {})


def test_invalid_json_raises_model_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    instance = model(client=client_for(handler))
    with pytest.raises(ModelUnavailableError, match="invalid JSON"):
        instance.generate([], 1, {})


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "   "},
        {"text": None},
        {"other": "field"},
        [1, 2, 3],
    ],
)
def test_missing_or_empty_text_field_raises_model_unavailable(payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    instance = model(client=client_for(handler))
    with pytest.raises(ModelUnavailableError, match="text"):
        instance.generate([], 1, {})


def test_server_reported_model_revision_replaces_unknown_when_unpinned():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "ok", "model_revision": "abc123def456"})

    instance = model(client=client_for(handler))
    assert instance.model_revision == "unknown"
    instance.generate_with_metadata([], 1, {})
    assert instance.model_revision == "abc123def456"


def test_explicit_model_revision_pin_is_never_overridden_by_the_server():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "ok", "model_revision": "server-reported-revision"})

    instance = ModalBatchModel(client=client_for(handler), environ=ENV, model_revision="operator-pinned")
    instance.generate_with_metadata([], 1, {})
    assert instance.model_revision == "operator-pinned"


def test_env_var_model_revision_pin_is_never_overridden_by_the_server():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "ok", "model_revision": "server-reported-revision"})

    instance = model(
        environ={**ENV, "MODEL_REVISION": "env-pinned"}, client=client_for(handler)
    )
    instance.generate_with_metadata([], 1, {})
    assert instance.model_revision == "env-pinned"


def test_missing_server_model_revision_leaves_unknown_unpinned():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "ok"})  # older-deployed endpoint

    instance = model(client=client_for(handler))
    instance.generate_with_metadata([], 1, {})
    assert instance.model_revision == "unknown"
