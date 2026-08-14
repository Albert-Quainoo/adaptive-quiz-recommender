"""Replay persisted attempts under a new configured BKT model version."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from app.bootstrap import AppSettings, build_controller
from bkt.schemas import BKTModelMetadata


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--bank", type=Path, required=True)
    root.add_argument("--database", type=Path, required=True)
    root.add_argument("--model", type=Path, required=True)
    root.add_argument("--metadata", type=Path, required=True)
    root.add_argument("--model-version", required=True)
    root.add_argument("--initial-mastery", type=float, required=True)
    return root


def main(argv=None) -> int:
    arguments = parser().parse_args(argv)
    metadata = json.loads(arguments.metadata.read_text(encoding="utf-8"))
    if metadata["model_version"] != arguments.model_version:
        raise ValueError("metadata model_version does not match --model-version")
    settings = AppSettings(
        database_path=arguments.database,
        approved_bank_path=arguments.bank,
        bkt_model_path=arguments.model,
        skills_path=Path("taxonomy/data/ai/skills.csv"),
        references_path=Path("taxonomy/data/ai/references.csv"),
        model_version=arguments.model_version,
        policy_version="recommendation-policy-v1",
        initial_mastery_probability=arguments.initial_mastery,
    )
    controller = build_controller(settings)
    controller.repository.save_model_metadata(
        BKTModelMetadata(
            model_version=arguments.model_version,
            fitted_at=datetime.fromisoformat(metadata["created_at"]),
            training_attempt_count=(
                metadata["synthetic_learner_count"]
                * metadata["opportunity_count"]
                * len(metadata["included_skill_ids"])
            ),
            skill_ids=metadata["included_skill_ids"],
        )
    )
    attempts = controller.repository.list_attempts()
    snapshots = controller.bkt_service.replay()
    print(
        json.dumps(
            {
                "model_version": arguments.model_version,
                "persisted_attempts": len(attempts),
                "replayed_snapshots": len(snapshots),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
