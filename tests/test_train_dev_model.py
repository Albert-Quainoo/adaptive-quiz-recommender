import json
from datetime import datetime, timezone

import pandas as pd
import pytest
from pyBKT.models import Model, Roster

from api.bank import BankItem
from api.schemas import QuizQuestion
from app.bootstrap import BootstrapError
from bkt.train_dev_model import (
    DEVELOPMENT_MODEL_DESCRIPTION,
    MODERATED_PILOT_PARAMETERS,
    OPPORTUNITY_COUNT,
    SYNTHETIC_LEARNER_COUNT,
    train_dev_model,
)


NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)
SKILLS = ["skill-a", "skill-b"]


def bank_item(item_id: str, skill_id: str) -> BankItem:
    return BankItem(
        item_id=item_id,
        skill_id=skill_id,
        provenance="hand_authored",
        question=QuizQuestion(
            question=f"Question for {skill_id}?",
            options=["Correct", "Wrong A", "Wrong B", "Wrong C"],
            correct_answer="Correct",
            explanation="A deterministic development-bank item.",
            concept=skill_id,
            difficulty="introductory",
        ),
    )


def write_bank(path, items):
    path.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
            for item in items
        ),
        encoding="utf-8",
    )


def fit_artifact(tmp_path, name="model"):
    bank_path = tmp_path / "approved-bank.jsonl"
    write_bank(
        bank_path,
        [
            bank_item("item-a-1", "skill-a"),
            bank_item("item-a-2", "skill-a"),
            bank_item("item-b-1", "skill-b"),
        ],
    )
    output_path = tmp_path / name / "bkt_dev_model.pkl"
    metadata_path = tmp_path / name / "bkt_dev_model.metadata.json"
    metadata = train_dev_model(
        bank_path,
        output_path,
        metadata_path,
        model_version="bkt-synthetic-v1",
        seed=42,
        clock=lambda: NOW,
    )
    return output_path, metadata_path, metadata


@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory):
    return fit_artifact(tmp_path_factory.mktemp("dev-model"))


def load_model(path):
    model = Model(parallel=False)
    model.load(str(path))
    return model


def test_identical_seed_and_bank_produce_equivalent_parameters(tmp_path):
    first_path, _, _ = fit_artifact(tmp_path, "first")
    second_path, _, _ = fit_artifact(tmp_path, "second")

    first_parameters = load_model(first_path).params().sort_index()
    second_parameters = load_model(second_path).params().sort_index()

    pd.testing.assert_frame_equal(first_parameters, second_parameters)


def test_all_bank_skills_are_included_and_metadata_is_written(trained_artifact):
    output_path, metadata_path, returned_metadata = trained_artifact

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata == returned_metadata
    assert metadata["included_skill_ids"] == SKILLS
    assert metadata["training_source"] == "synthetic"
    assert metadata["synthetic_learner_count"] == SYNTHETIC_LEARNER_COUNT
    assert metadata["opportunity_count"] == OPPORTUNITY_COUNT
    assert metadata["model_artifact_path"] == str(output_path)
    assert metadata["description"] == DEVELOPMENT_MODEL_DESCRIPTION
    assert metadata["created_at"] == NOW.isoformat()
    assert metadata["pybkt_version"]


def test_saved_model_loads_and_roster_updates_every_skill(trained_artifact):
    output_path, _, _ = trained_artifact
    model = load_model(output_path)

    assert model.fit_model is not None
    assert set(model.params().index.get_level_values("skill")) == set(SKILLS)
    for skill_id in SKILLS:
        roster = Roster(students=["learner"], skills=skill_id, model=model)
        before = roster.get_mastery_prob(skill_id, "learner")
        roster.update_state(skill_id, "learner", 1)
        after = roster.get_mastery_prob(skill_id, "learner")
        assert 0.0 <= before <= 1.0
        assert 0.0 <= after <= 1.0
        assert after >= before


def test_empty_approved_bank_is_rejected(tmp_path):
    bank_path = tmp_path / "empty-bank.jsonl"
    bank_path.write_text("", encoding="utf-8")

    with pytest.raises(BootstrapError, match="approved learner-facing bank is empty"):
        train_dev_model(
            bank_path,
            tmp_path / "model.pkl",
            tmp_path / "metadata.json",
            model_version="bkt-synthetic-v1",
            seed=42,
        )


def test_moderated_profile_is_fixed_and_has_gradual_trajectories(tmp_path):
    bank_path = tmp_path / "approved-bank.jsonl"
    write_bank(
        bank_path,
        [bank_item("item-a", "skill-a"), bank_item("item-b", "skill-b")],
    )
    output_path = tmp_path / "moderated.pkl"
    metadata_path = tmp_path / "moderated.metadata.json"

    metadata = train_dev_model(
        bank_path,
        output_path,
        metadata_path,
        model_version="bkt-synthetic-v4",
        seed=42,
        parameter_profile="moderated-pilot",
        clock=lambda: NOW,
    )
    model = load_model(output_path)

    assert metadata["parameter_profile"] == "moderated-pilot"
    assert metadata["parameters"] == MODERATED_PILOT_PARAMETERS
    for skill_id in SKILLS:
        parameters = model.coef_[skill_id]
        assert parameters["prior"] == pytest.approx(0.20)
        assert parameters["learns"][0] == pytest.approx(0.04)
        assert parameters["guesses"][0] == pytest.approx(0.25)
        assert parameters["slips"][0] == pytest.approx(0.20)
        roster = Roster(students=["learner"], skills=skill_id, model=model)
        roster.update_state(skill_id, "learner", 0)
        after_incorrect = roster.get_mastery_prob(skill_id, "learner")
        roster.update_state(skill_id, "learner", 1)
        after_correct = roster.get_mastery_prob(skill_id, "learner")
        assert after_incorrect < 0.20
        assert 0.20 < after_correct < 0.30

        fresh_roster = Roster(students=["fresh-learner"], skills=skill_id, model=model)
        fresh_roster.update_state(skill_id, "fresh-learner", 1)
        after_first_correct = fresh_roster.get_mastery_prob(skill_id, "fresh-learner")
        fresh_roster.update_state(skill_id, "fresh-learner", 1)
        after_second_correct = fresh_roster.get_mastery_prob(skill_id, "fresh-learner")
        assert after_first_correct < 0.55
        assert after_second_correct < 0.80
    MODERATED_PILOT_PARAMETERS,
