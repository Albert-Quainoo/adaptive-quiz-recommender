from collections.abc import Sequence

import pandas as pd

from bkt.schemas import AttemptEvent


PYBKT_COLUMNS = ["order_id", "user_id", "skill_name", "correct"]


def attempts_to_dataframe(attempts: Sequence[AttemptEvent]) -> pd.DataFrame:
    """Convert attempts to pyBKT's canonical columns in explicit order."""

    return pd.DataFrame(
        [
            {
                "order_id": attempt.attempt_order,
                "user_id": attempt.learner_id,
                "skill_name": attempt.skill_id,
                "correct": int(attempt.correct),
            }
            for attempt in sorted(
                attempts,
                key=lambda item: (
                    item.attempt_order,
                    item.occurred_at,
                    item.attempt_id,
                ),
            )
        ],
        columns=PYBKT_COLUMNS,
    )


class PyBKTAdapter:
    """Object-form adapter for dependency injection and extension points."""

    def to_dataframe(self, attempts: Sequence[AttemptEvent]) -> pd.DataFrame:
        return attempts_to_dataframe(attempts)
