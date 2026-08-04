from pydantic import BaseModel
from typing import Literal

from api.schemas import QuizQuestion

provenance_kind = Literal[
    "generated",
    "templated",
    "hand_authored",
]


class BankItem(BaseModel):
    question: QuizQuestion
    provenance: provenance_kind
    skill_id: str | None = None
