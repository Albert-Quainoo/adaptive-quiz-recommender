"""Assert that every card's content-hash badge in three-course-content-review.html
matches question_content_hash() recomputed from the exact approved-bank JSONL on disk --
the check that proves the packet cannot silently drift from the bank.

Run with: python -m scripts.verify_packet_hashes
"""

import json
import re
from pathlib import Path

from api.schemas import QuizQuestion
from authoring.grounded_review import question_content_hash

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKET_PATH = REPO_ROOT / "docs" / "three-course-content-review.html"

BANK_PATHS = [
    REPO_ROOT / "outputs/approved_banks/dsa-approved-bank-28-v1.jsonl",
    REPO_ROOT / "outputs/approved_banks/linear-algebra-approved-bank-24-v1.jsonl",
    REPO_ROOT / "outputs/approved_banks/database-systems-approved-bank-28-v1.jsonl",
]

CARD_PATTERN = re.compile(
    r'Bank item_id:</strong> <span class="mono">([^<]+)</span>.*?content hash \(sha256\):</strong> '
    r'<span class="mono">([0-9a-f]{64})</span>',
    re.DOTALL,
)


def main() -> None:
    bank_items: dict[str, dict] = {}
    for path in BANK_PATHS:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            bank_items[record["item_id"]] = record

    packet_text = PACKET_PATH.read_text(encoding="utf-8")
    matches = CARD_PATTERN.findall(packet_text)
    if not matches:
        raise SystemExit("no card hash badges found in packet")

    failures = []
    for item_id, packet_hash in matches:
        bank_record = bank_items.get(item_id)
        if bank_record is None:
            failures.append(f"{item_id}: not found in any approved bank")
            continue
        actual_hash = question_content_hash(QuizQuestion(**bank_record["question"]))
        if actual_hash != packet_hash:
            failures.append(f"{item_id}: packet hash {packet_hash} != bank hash {actual_hash}")

    print(f"checked {len(matches)} card(s) against {len(bank_items)} bank item(s)")
    if failures:
        for failure in failures:
            print("FAIL:", failure)
        raise SystemExit(1)
    print("all packet card hashes match the exact approved-bank artifacts byte-for-byte")


if __name__ == "__main__":
    main()
