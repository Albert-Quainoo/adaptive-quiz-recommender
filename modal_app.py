"""Modal deployment of the Llama inference used by content-replenishment
question generation.

    modal deploy modal_app.py    # deploy; prints the endpoint URL
    modal run modal_app.py       # send one known prompt directly (no HTTP,
                                  # no proxy auth) to compare against the
                                  # Kaggle baseline before wiring the pipeline

Reuses api/model_loader.py's model/tokenizer loading and the same generation
mechanics as authoring/replenishment/worker.py:LlamaBatchModel -- only where
that inference runs differs. The HTTP contract this endpoint serves is
documented in README.md's "Modal inference provider" section and consumed by
authoring/replenishment/modal_inference.py:ModalBatchModel.
"""

import os
from pathlib import Path

import modal
from pydantic import BaseModel

# Read locally (on the machine running `modal deploy`), not inside the
# container -- MODEL_REPOSITORY isn't sensitive, so it is baked into the
# image as a plain env var rather than a Secret. This mirrors
# api/model_loader.py:get_model_id() without importing that module (and
# therefore torch) into the local deploy-time process.
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
MODEL_REPOSITORY = os.environ.get("MODEL_REPOSITORY")
if not MODEL_REPOSITORY:
    raise RuntimeError(
        "MODEL_REPOSITORY is missing from .env; set it before running `modal deploy`."
    )

GPU = "L4"

app = modal.App("adaptive-quiz-llama-inference")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "fastapi[standard]",
    )
    .env({"MODEL_REPOSITORY": MODEL_REPOSITORY})
    .add_local_python_source("api")
)

# huggingface-secret must expose HF_TOKEN -- required_keys makes a
# misconfigured secret fail at deploy/attach time instead of at first request.
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])


class GenerateRequest(BaseModel):
    messages: list[dict[str, str]]
    seed: int
    generation_parameters: dict = {}


@app.cls(image=image, gpu=GPU, secrets=[hf_secret], scaledown_window=300)
class LlamaInference:
    @modal.enter()
    def load(self) -> None:
        from api.model_loader import load_model, load_tokenizer

        self.tokenizer = load_tokenizer()
        self.model = load_model()

    def _generate(
        self, messages: list[dict[str, str]], seed: int, generation_parameters: dict
    ) -> dict:
        import torch

        max_new_tokens = generation_parameters.get("max_new_tokens", 800)

        torch.manual_seed(seed)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        input_tokens = inputs["input_ids"].shape[1]
        generated_tokens = output[0, input_tokens:]
        output_tokens = generated_tokens.shape[0]
        # Greedy decoding stops either because it emitted eos_token_id (a genuine
        # "stop") or because it hit max_new_tokens first (a "length" truncation,
        # possibly mid-token/mid-field for a structured-output caller) -- token count
        # alone can't distinguish these when the model happens to emit exactly
        # max_new_tokens tokens and the last one is also eos, so check the token id.
        if output_tokens > 0 and generated_tokens[-1].item() == self.tokenizer.eos_token_id:
            finish_reason = "stop"
        elif output_tokens >= max_new_tokens:
            finish_reason = "length"
        else:
            finish_reason = "unknown"
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # The exact commit hash transformers resolved for this model from the Hub --
        # a real, pinned identifier, distinct from a hand-maintained MODEL_REVISION
        # env var. None if this version of transformers doesn't set it.
        model_revision = getattr(self.model.config, "_commit_hash", None)
        return {
            "text": text,
            "finish_reason": finish_reason,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "max_new_tokens": max_new_tokens,
            "model_revision": model_revision,
        }

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate(self, request: GenerateRequest) -> dict:
        return self._generate(
            request.messages, request.seed, request.generation_parameters
        )

    @modal.method()
    def generate_direct(
        self, messages: list[dict[str, str]], seed: int, generation_parameters: dict
    ) -> str:
        """Bypasses HTTP and proxy auth -- for `modal run modal_app.py` only."""
        return self._generate(messages, seed, generation_parameters)["text"]


@app.local_entrypoint()
def main() -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "In one sentence, what does a heuristic function estimate in search?",
        },
    ]
    result = LlamaInference().generate_direct.remote(messages, 42, {"max_new_tokens": 200})
    print(result)
