from collections.abc import Sequence

import pandas as pd

from bkt.schemas import AttemptEvent


PYBKT_COLUMNS = ["order_id", "user_id", "skill_name", "correct"]


def attempts_to_dataframe(attempts: Sequence[AttemptEvent]) -> pd.DataFrame:
    """Convert already ordered attempts to pyBKT's canonical columns."""

    return pd.DataFrame(
        [
            {
                "order_id": order_id,
                "user_id": attempt.learner_id,
                "skill_name": attempt.skill_id,
                "correct": int(attempt.correct),
            }
            for order_id, attempt in enumerate(attempts)
        ],
        columns=PYBKT_COLUMNS,
    )


class PyBKTAdapter:
    """Object-form adapter for dependency injection and extension points."""

    def to_dataframe(self, attempts: Sequence[AttemptEvent]) -> pd.DataFrame:
        return attempts_to_dataframe(attempts)
