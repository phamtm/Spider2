"""Explicit default policy surface for sol01 runtime and heuristics."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SolverPolicy:
    """Retry and recovery budgets for one solver run."""

    initial_candidates: int = 3
    max_attempts: int = 4
    semantic_repairs: int = 1

    def __post_init__(self) -> None:
        if self.initial_candidates < 1:
            raise ValueError("initial_candidates must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.semantic_repairs < 0:
            raise ValueError("semantic_repairs must be zero or positive")
        if self.initial_candidates > self.max_attempts:
            raise ValueError("initial_candidates must not exceed max_attempts")

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-ready trace payload."""

        return asdict(self)


@dataclass(frozen=True)
class RuntimeProfile:
    """Default LLM/runtime profile for sol01."""

    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "deepseek/deepseek-v4-pro"
    provider_only: str = "deepseek"
    allow_fallbacks: bool = False
    concurrency: int = 4


@dataclass(frozen=True)
class SchemaContextPolicy:
    """Defaults for schema planning and prompt context sizing."""

    version: str = "schema_context_v1"
    family_similarity_threshold: float = 0.82
    max_linked_doc_chars: int = 6000
    max_schema_prompt_chars: int = 24000


@dataclass(frozen=True)
class PromptBudgetPolicy:
    """How planner prompts shrink when they exceed the total budget."""

    shrink_order: tuple[str, ...] = ("evidence", "docs")
    shrink_overflow_buffer_chars: int = 256
    max_shrink_rounds: int = 12
    evidence_line_max_chars: int = 500


@dataclass(frozen=True)
class SchemaRenderPolicy:
    """Bounds for planner-visible and SQL-visible schema rendering."""

    family_members_in_prompt: int = 12
    family_members_to_expand: int = 64
    variant_columns_in_prompt: int = 12
    schema_context_evidence_lines: int = 8
    table_columns_in_prompt: int = 30
    sample_literal_chars: int = 80


@dataclass(frozen=True)
class FilterGroundingPolicy:
    """Caps for empty-result filter grounding probes."""

    max_probe_targets: int = 4
    fallback_string_columns: int = 2
    probe_match_limit: int = 5


@dataclass(frozen=True)
class RecoverySignalPolicy:
    """Signals that trigger schema-aware recovery decisions."""

    priority_order: tuple[str, ...] = ("schema", "sql", "semantic")
    execution_error_preview_chars: int = 300


@dataclass(frozen=True)
class EvalDatasetPolicy:
    """Dataset metadata for official-eval summaries."""

    name: str = "spider2-snow"
    default_task_count: int = 547


@dataclass(frozen=True)
class SchemaContextEvalPolicy:
    """Bounds for offline schema-context evaluation reports."""

    object_cutoff: int = 12
    failure_evidence_limit: int = 5
    failure_limit: int = 20
    prompt_win_limit: int = 20
    prompt_win_threshold: float = 0.25


DEFAULT_SOLVER_POLICY = SolverPolicy()
DEFAULT_RUNTIME_PROFILE = RuntimeProfile()
DEFAULT_SCHEMA_CONTEXT_POLICY = SchemaContextPolicy()
DEFAULT_PROMPT_BUDGET_POLICY = PromptBudgetPolicy()
DEFAULT_SCHEMA_RENDER_POLICY = SchemaRenderPolicy()
DEFAULT_FILTER_GROUNDING_POLICY = FilterGroundingPolicy()
DEFAULT_RECOVERY_SIGNAL_POLICY = RecoverySignalPolicy()
DEFAULT_EVAL_DATASET_POLICY = EvalDatasetPolicy()
DEFAULT_SCHEMA_CONTEXT_EVAL_POLICY = SchemaContextEvalPolicy()
