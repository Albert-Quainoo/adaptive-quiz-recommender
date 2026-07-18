import os
from functools import lru_cache
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)


def get_model_id() -> str:
    model_id = os.getenv("MODEL_REPOSITORY")

    if not model_id:
        raise RuntimeError("MODEL_REPOSITORY is missing from the environment")

    return model_id


def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError("HF_TOKEN is missing from the environment")

    return token


@lru_cache(maxsize=1)
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        get_model_id(),
        token=get_hf_token(),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


@lru_cache(maxsize=1)
def load_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    return AutoModelForCausalLM.from_pretrained(
        get_model_id(),
        token=get_hf_token(),
        device_map="auto",
        quantization_config=quantization_config,
        dtype=torch.float16,
    )
