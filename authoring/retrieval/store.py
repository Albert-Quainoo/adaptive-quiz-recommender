"""Where candidates live between CLI runs.

Review happens at human pace: a candidate is retrieved in one run, read later,
and approved later still. A JSON file keyed by candidate_id is enough for that,
and being deterministic on disk means the review history diffs cleanly.

Retrieving again never disturbs a decision already made - a candidate the store
already holds keeps its status, its reviewer and its note.
"""

import json
from collections.abc import Iterable
from pathlib import Path

from authoring.retrieval.models import ReferenceCandidate, ReviewStatus


class StoreError(ValueError):
    pass


class CandidateStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> list[ReferenceCandidate]:
        if not self.path.exists():
            return []

        payload = json.loads(self.path.read_text(encoding="utf-8"))

        return [ReferenceCandidate.model_validate(record) for record in payload]

    def save(self, candidates: Iterable[ReferenceCandidate]) -> None:
        records = [
            candidate.model_dump(mode="json")
            for candidate in sorted(candidates, key=lambda item: item.candidate_id)
        ]

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def add(self, candidates: Iterable[ReferenceCandidate]) -> list[ReferenceCandidate]:
        """Merge a retrieval run in, keeping every decision already recorded."""
        held = {candidate.candidate_id: candidate for candidate in self.load()}
        added = []

        for candidate in candidates:
            if candidate.candidate_id in held:
                continue

            held[candidate.candidate_id] = candidate
            added.append(candidate)

        self.save(held.values())

        return added

    def get(self, candidate_id: str) -> ReferenceCandidate:
        for candidate in self.load():
            if candidate.candidate_id == candidate_id:
                return candidate

        raise StoreError(f"{candidate_id} is not in {self.path}.")

    def replace(self, candidate: ReferenceCandidate) -> ReferenceCandidate:
        held = {item.candidate_id: item for item in self.load()}

        if candidate.candidate_id not in held:
            raise StoreError(f"{candidate.candidate_id} is not in {self.path}.")

        held[candidate.candidate_id] = candidate
        self.save(held.values())

        return candidate

    def with_status(self, status: ReviewStatus) -> list[ReferenceCandidate]:
        return [
            candidate for candidate in self.load() if candidate.review_status == status
        ]
