from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from bkt import AttemptEvent, BKTModelMetadata, MasterySnapshot


NOW = datetime(2026, 8, 6, tzinfo=timezone.utc)


def test_bkt_schemas_normalise_identifiers_and_bound_mastery():
    attempt = AttemptEvent(
        attempt_id=" a-1 ",
        presentation_id=" presentation-1 ",
        attempt_order=1,
        learner_id=" learner-1 ",
        item_id=" item-1 ",
        skill_id=" skill-1 ",
        selected_option_id=" option-1 ",
        correct=True,
        occurred_at=NOW,
    )
    assert (
        attempt.attempt_id,
        attempt.presentation_id,
        attempt.learner_id,
        attempt.item_id,
        attempt.skill_id,
        attempt.selected_option_id,
    ) == (
        "a-1",
        "presentation-1",
        "learner-1",
        "item-1",
        "skill-1",
        "option-1",
    )

    with pytest.raises(ValidationError):
        MasterySnapshot(
            learner_id="learner-1",
            skill_id="skill-1",
            mastery_probability=1.1,
            attempt_count=1,
            model_version="v1",
            source_attempt_id="a-1",
        )


def test_model_metadata_requires_unique_skills():
    with pytest.raises(ValidationError):
        BKTModelMetadata(
            model_version="v1",
            training_attempt_count=2,
            skill_ids=["skill-1", "skill-1"],
        )
