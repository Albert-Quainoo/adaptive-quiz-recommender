from pydantic import BaseModel, Field
from typing import Literal

from api.schemas import QuizQuestion

provenance_kind = Literal[
    "generated",
    "templated",
    "hand_authored",
]


class BankItem(BaseModel):
    item_id: str | None = Field(default=None, min_length=1)
    question: QuizQuestion
    provenance: provenance_kind
    skill_id: str | None = None
