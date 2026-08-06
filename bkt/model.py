from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

import pandas as pd

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
    ) -> None:
        self._model = model if model is not None else self._create_model(seed, num_fits)
        self.model_version = model_version.strip()
        if not self.model_version:
            raise ValueError("model_version is required")
        self._adapter = adapter or PyBKTAdapter()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._fitted = False

    @staticmethod
    def _create_model(seed: int, num_fits: int) -> Any:
        try:
            from pyBKT.models import Model
        except ImportError as exc:
            raise RuntimeError(
                "pyBKT is required to construct BKTModel without an injected model"
            ) from exc
        return Model(seed=seed, num_fits=num_fits)

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
        """Return mastery after the final attempt using pyBKT's own update logic."""

        self._require_fitted()
        if not attempts:
            raise ValueError("at least one attempt is required for a mastery update")

        frame = self._adapter.to_dataframe(attempts)
        final_attempt = attempts[-1]
        probe = pd.DataFrame(
            [
                {
                    "order_id": len(frame),
                    "user_id": final_attempt.learner_id,
                    "skill_name": final_attempt.skill_id,
                    "correct": -1,
                }
            ]
        )
        predictions = self._model.predict(
            data=pd.concat([frame, probe], ignore_index=True)
        )
        if "state_predictions" not in predictions:
            raise ValueError("pyBKT predictions did not include state_predictions")

        probability = float(predictions.iloc[-1]["state_predictions"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError("pyBKT returned an invalid mastery probability")
        return probability

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("BKT model must be fitted before prediction")
