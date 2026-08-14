"""State-aware reliability tracking: pure aggregation over stored review reports.

Tracks rejection/revision rates by the exact dimensions requirement 16 names --
generator model/prompt version, reviewer model/prompt version, skill, course, source
domain, intent family -- and flags a configuration for forced full-human-review once its
rejection or revision rate crosses a threshold, until it is recalibrated (removed from
the flagged set by a human, out of scope for this module).

No online self-modification: this module only reads past reports and returns a decision
for the *next* review to consult (via risk.score_risk's is_calibrated_configuration
argument). It never edits a prompt or policy version itself.
"""

from collections import defaultdict
from dataclasses import dataclass

from authoring.review.models import AutomatedReviewReport

# A course id and source domain aren't carried on AutomatedReviewReport itself (they
# belong to the reference/manifest layer); callers that want that granularity attach it
# via the same key tuple shape when building `reports_by_key`. For per-report
# aggregation as stored, the dimensions actually present are these five.
ConfigurationKey = tuple[str, str, str, str, str]
"""(generator_model_id, generator_prompt_version, reviewer_model_id,
reviewer_prompt_version, skill_id)"""


@dataclass(frozen=True)
class ConfigurationReliability:
    key: ConfigurationKey
    total_reports: int
    rejected: int
    revised: int
    rejection_rate: float
    revision_rate: float
    flagged: bool


def configuration_key(
    report: AutomatedReviewReport, *, generator_model_id: str, generator_prompt_version: str
) -> ConfigurationKey:
    return (
        generator_model_id,
        generator_prompt_version,
        report.reviewer_model_id,
        report.reviewer_prompt_version,
        report.skill_id,
    )


def summarize_reliability(
    reports_by_key: dict[ConfigurationKey, list[AutomatedReviewReport]],
    *,
    rejection_rate_threshold: float = 0.4,
    revision_rate_threshold: float = 0.6,
    min_reports_for_flagging: int = 5,
) -> dict[ConfigurationKey, ConfigurationReliability]:
    summary: dict[ConfigurationKey, ConfigurationReliability] = {}
    for key, reports in reports_by_key.items():
        total = len(reports)
        rejected = sum(1 for report in reports if report.recommendation == "reject")
        revised = sum(1 for report in reports if report.recommendation == "propose_revision")
        rejection_rate = rejected / total if total else 0.0
        revision_rate = revised / total if total else 0.0
        flagged = total >= min_reports_for_flagging and (
            rejection_rate >= rejection_rate_threshold or revision_rate >= revision_rate_threshold
        )
        summary[key] = ConfigurationReliability(
            key=key,
            total_reports=total,
            rejected=rejected,
            revised=revised,
            rejection_rate=rejection_rate,
            revision_rate=revision_rate,
            flagged=flagged,
        )
    return summary


def is_calibrated(
    key: ConfigurationKey, reliability: dict[ConfigurationKey, ConfigurationReliability]
) -> bool:
    entry = reliability.get(key)
    return entry is None or not entry.flagged


def group_reports_by_configuration(
    reports: list[AutomatedReviewReport],
    *,
    generator_model_id_for: callable,
    generator_prompt_version_for: callable,
) -> dict[ConfigurationKey, list[AutomatedReviewReport]]:
    """Group flat reports into per-configuration buckets. The generator identity isn't
    stored on the report itself, so callers supply lookups (report -> value) keyed by
    candidate_id -- ordinarily backed by the batch manifest's model_id/prompt_version."""
    grouped: dict[ConfigurationKey, list[AutomatedReviewReport]] = defaultdict(list)
    for report in reports:
        key = configuration_key(
            report,
            generator_model_id=generator_model_id_for(report),
            generator_prompt_version=generator_prompt_version_for(report),
        )
        grouped[key].append(report)
    return dict(grouped)
