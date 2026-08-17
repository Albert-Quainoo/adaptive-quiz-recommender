"""Bounded per-cycle budget for the scheduled replenishment workflow.

Caps how much of one scheduled run is allowed to spend: how many brand-new
skill episodes it may start, how many model/reviewer calls it may make, and
an estimated dollar ceiling. Every check happens between job-processing
ticks, never mid-tick, so a run that stops here always stops at a point the
worker's own resumable, stage-based design (authoring/replenishment/worker.py)
already handles cleanly -- the next run resumes from exactly where this one
left off.

Cost is a coarse, configurable per-call estimate, not real billing: nothing
in the pipeline today plumbs token-level accounting out of generate_batch's
question-generation calls (only the reviewer's ModelBackedContentReviewer
captures GenerationOutcome token counts), so per-call estimates are the
honest level of precision to report here.
"""

from dataclasses import dataclass, field

# Which job_type a claimed job carries indicates which paid call, if any,
# process_job() is about to make for that tick -- see worker.py's
# process_job() dispatch table, which this mirrors read-only.
GENERATION_JOB_TYPES = frozenset({"generate_questions", "validate_questions", "automated_revision"})
REVIEW_JOB_TYPES = frozenset({"automated_review"})
SEARCH_JOB_TYPES = frozenset({"replenish_skill", "retrieve_references"})


@dataclass(frozen=True)
class CycleBudgetConfig:
    max_new_candidates: int = 3
    max_generation_calls: int = 20
    max_cost_usd: float = 5.0
    cost_per_generation_call_usd: float = 0.05
    cost_per_review_call_usd: float = 0.02
    cost_per_search_call_usd: float = 0.0
    # Safety valve independent of the above: bounds worst-case run length
    # (e.g. a long chain of resumed waiting jobs with zero paid calls)
    # so a scheduled run can never hang a CI job indefinitely.
    max_ticks: int = 500

    def __post_init__(self) -> None:
        if self.max_new_candidates <= 0:
            raise ValueError("max_new_candidates must be positive")
        if self.max_generation_calls <= 0:
            raise ValueError("max_generation_calls must be positive")
        if self.max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive")
        if self.max_ticks <= 0:
            raise ValueError("max_ticks must be positive")
        for name in (
            "cost_per_generation_call_usd",
            "cost_per_review_call_usd",
            "cost_per_search_call_usd",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class CycleBudgetTracker:
    """Mutable run-scoped counters. One instance per scheduled-run invocation."""

    config: CycleBudgetConfig
    ticks: int = 0
    generation_calls: int = 0
    review_calls: int = 0
    search_calls: int = 0
    new_candidate_job_ids: set[str] = field(default_factory=set)

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.generation_calls * self.config.cost_per_generation_call_usd
            + self.review_calls * self.config.cost_per_review_call_usd
            + self.search_calls * self.config.cost_per_search_call_usd
        )

    def exhausted(self) -> str | None:
        """A human-readable reason this budget will accept no further ticks
        this run, or None if there is still room."""
        if self.ticks >= self.config.max_ticks:
            return f"tick safety limit reached ({self.config.max_ticks})"
        if self.generation_calls + self.review_calls >= self.config.max_generation_calls:
            return f"generation/review call cap reached ({self.config.max_generation_calls})"
        if self.estimated_cost_usd >= self.config.max_cost_usd:
            return f"estimated cost cap reached (${self.config.max_cost_usd:.2f})"
        return None

    def would_start_new_candidate_beyond_cap(self, job_id: str, *, is_new: bool) -> bool:
        """True when claiming this job would begin a brand-new skill episode
        (never touched earlier in this run) while the new-candidate cap is
        already spent. Resuming a job already counted this run, or any job
        that existed before this run started (is_new=False), is unaffected --
        only *starting* a new episode is capped."""
        return (
            is_new
            and job_id not in self.new_candidate_job_ids
            and len(self.new_candidate_job_ids) >= self.config.max_new_candidates
        )

    def record_tick(self, *, job_type: str, job_id: str, is_new: bool) -> None:
        self.ticks += 1
        if job_type in GENERATION_JOB_TYPES:
            self.generation_calls += 1
        elif job_type in REVIEW_JOB_TYPES:
            self.review_calls += 1
        elif job_type in SEARCH_JOB_TYPES:
            self.search_calls += 1
        if is_new:
            self.new_candidate_job_ids.add(job_id)

    def to_dict(self) -> dict:
        return {
            "ticks": self.ticks,
            "generation_calls": self.generation_calls,
            "review_calls": self.review_calls,
            "search_calls": self.search_calls,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "new_candidates_started": len(self.new_candidate_job_ids),
            "max_new_candidates": self.config.max_new_candidates,
            "max_generation_calls": self.config.max_generation_calls,
            "max_cost_usd": self.config.max_cost_usd,
        }
