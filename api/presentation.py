"""Deterministic, reconstructable option ordering at presentation time."""

import hashlib
import random

from pydantic import BaseModel, Field, model_validator

from api.bank import BankItem


class PresentedOption(BaseModel):
    option_id: str = Field(min_length=1)
    value: str


class QuestionPresentation(BaseModel):
    presentation_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    presentation_seed: int = Field(ge=0)
    presented_options: list[PresentedOption] = Field(min_length=1)

    @model_validator(mode="after")
    def option_ids_and_values_are_distinct(self) -> "QuestionPresentation":
        option_ids = [option.option_id for option in self.presented_options]
        values = [option.value for option in self.presented_options]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("presented option ids must be distinct")
        if len(values) != len(set(values)):
            raise ValueError("presented option values must be distinct")
        return self


def _stable_integer(*values: str) -> int:
    digest = hashlib.sha256("\0".join(values).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def derive_presentation_seed(
    item_id: str,
    learner_id: str,
    attempt_id: str,
) -> int:
    """Derive a reproducible local-RNG seed from stable presentation inputs."""
    values = [value.strip() for value in (item_id, learner_id, attempt_id)]
    if any(not value for value in values):
        raise ValueError("item_id, learner_id and attempt_id are required")
    return _stable_integer(*values)


def option_id(item_id: str, option_value: str) -> str:
    digest = hashlib.sha256(f"{item_id}\0{option_value}".encode("utf-8")).hexdigest()
    return f"opt-{digest[:16]}"


def presentation_from_seed(item: BankItem, presentation_seed: int) -> QuestionPresentation:
    """Shuffle a copy with an isolated RNG, leaving the BankItem untouched."""
    if not item.item_id:
        raise ValueError("a presented BankItem needs an item_id")
    options = list(item.question.options)
    random.Random(presentation_seed).shuffle(options)
    presentation_id = hashlib.sha256(
        f"{item.item_id}\0{presentation_seed}".encode("utf-8")
    ).hexdigest()[:24]
    return QuestionPresentation(
        presentation_id=f"presentation-{presentation_id}",
        item_id=item.item_id,
        presentation_seed=presentation_seed,
        presented_options=[
            PresentedOption(option_id=option_id(item.item_id, value), value=value)
            for value in options
        ],
    )


def present_bank_item(
    item: BankItem,
    *,
    learner_id: str,
    attempt_id: str,
) -> QuestionPresentation:
    if not item.item_id:
        raise ValueError("a presented BankItem needs an item_id")
    seed = derive_presentation_seed(item.item_id, learner_id, attempt_id)
    return presentation_from_seed(item, seed)


def score_response(
    item: BankItem,
    presentation: QuestionPresentation,
    *,
    submitted_value: str | None = None,
    submitted_option_id: str | None = None,
) -> bool:
    """Score one exact option value or its stable presentation option ID."""
    if item.item_id != presentation.item_id:
        raise ValueError("presentation does not belong to this BankItem")
    if (submitted_value is None) == (submitted_option_id is None):
        raise ValueError("submit exactly one option value or option id")
    expected = presentation_from_seed(item, presentation.presentation_seed)
    if expected != presentation:
        raise ValueError("presentation order or identity does not match its seed")
    if submitted_option_id is not None:
        selected = next(
            (
                option.value
                for option in presentation.presented_options
                if option.option_id == submitted_option_id
            ),
            None,
        )
        if selected is None:
            raise ValueError("submitted option id is not in this presentation")
    else:
        selected = submitted_value
        if selected not in {option.value for option in presentation.presented_options}:
            raise ValueError("submitted option value is not in this presentation")
    return selected == item.question.correct_answer
