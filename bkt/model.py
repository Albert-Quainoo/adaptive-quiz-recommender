from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from pyBKT.models import Roster

from bkt.adapter import PyBKTAdapter
from bkt.schemas import AttemptEvent, BKTModelMetadata


class BKTModel:
    """Thin wrapper around pyBKT fitting, prediction, and mastery updates."""

    def __init__(
        self,
        model: Any | None = None,
        *,
        model_version: str = "unversioned",
        seed: int = 42,
        num_fits: int = 1,
        adapter: PyBKTAdapter | None = None,
        clock: Callable[[], datetime] | None = None,
        fitted: bool = False,
    ) -> None:
        self._model = model if model is not None else self._create_model(seed, num_fits)
        self.model_version = model_version.strip()
        if not self.model_version:
            raise ValueError("model_version is required")
        self._adapter = adapter or PyBKTAdapter()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._fitted = fitted

    @staticmethod
    def _create_model(seed: int, num_fits: int) -> Any:
        try:
            from pyBKT.models import Model
        except ImportError as exc:
            raise RuntimeError(
                "pyBKT is required to construct BKTModel without an injected model"
            ) from exc
        return Model(seed=seed, num_fits=num_fits, parallel=False)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, attempts: Sequence[AttemptEvent]) -> BKTModelMetadata:
        if not attempts:
            raise ValueError("at least one attempt is required to fit a BKT model")

        frame = self._adapter.to_dataframe(attempts)
        self._model.fit(data=frame)
        self._fitted = True
        return BKTModelMetadata(
            model_version=self.model_version,
            fitted_at=self._clock(),
            training_attempt_count=len(attempts),
            skill_ids=sorted({attempt.skill_id for attempt in attempts}),
        )

    def predict(self, attempts: Sequence[AttemptEvent]) -> pd.DataFrame:
        self._require_fitted()
        if not attempts:
            raise ValueError("at least one attempt is required for prediction")
        return self._model.predict(data=self._adapter.to_dataframe(attempts))

    def update_mastery(self, attempts: Sequence[AttemptEvent]) -> float:
        self._require_fitted()

        if not attempts:
            raise ValueError("at least one attempt is required")

        learner_ids = {attempt.learner_id for attempt in attempts}
        skill_ids = {attempt.skill_id for attempt in attempts}

        if len(learner_ids) != 1 or len(skill_ids) != 1:
            raise ValueError(
                "mastery history must contain exactly one learner and one skill"
            )

        ordered_attempts = sorted(
            attempts,
            key=lambda item: (
                item.attempt_order,
                item.occurred_at,
                item.attempt_id,
            ),
        )
        learner_id = ordered_attempts[0].learner_id
        skill_id = ordered_attempts[0].skill_id

        roster = Roster(
            students=[learner_id],
            skills=skill_id,
            model=self._model,
        )

        for attempt in ordered_attempts:
            roster.update_state(skill_id, learner_id, int(attempt.correct))

        probability = float(roster.get_mastery_prob(skill_id, learner_id))

        if not 0.0 <= probability <= 1.0:
            raise ValueError("pyBKT returned an invalid mastery probability")

        return probability

    def get_parameters(self) -> pd.DataFrame:
        self._require_fitted()
        return self._model.params()

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("BKT model must be fitted before prediction")
