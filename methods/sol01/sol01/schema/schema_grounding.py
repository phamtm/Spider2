"""Ground requested terms against exact selected-table metadata."""

from __future__ import annotations

from collections.abc import Sequence

from sol01.models import (
    Intent,
    SchemaGrounding,
    SchemaGroundingBinding,
    TableSchema,
    UnresolvedSchemaTerm,
)


def grounding_targets(intent: Intent) -> list[dict[str, object]]:
    """Return distinct requested terms that should bind to selected-table schema."""

    targets: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    buckets = [
        ("metric", intent.metrics, True),
        ("filter", intent.filters, True),
        ("native_term", intent.native_value_terms, True),
        ("entity", intent.entities, False),
        ("order", intent.requested_ordering, False),
    ]
    for binding_kind, terms, essential in buckets:
        for raw_term in terms:
            term = str(raw_term).strip()
            if not term:
                continue
            key = (binding_kind, term)
            if key in seen:
                continue
            seen.add(key)
            targets.append(
                {
                    "requested_term": term,
                    "binding_kind": binding_kind,
                    "essential": essential,
                }
            )
    return targets


def sanitize_schema_grounding(
    grounding: SchemaGrounding,
    *,
    available_tables: Sequence[str],
    table_schemas: dict[str, TableSchema],
    requested_terms: Sequence[dict[str, object]],
) -> tuple[SchemaGrounding, dict[str, object]]:
    """Keep only bindings to exact selected-table columns and record violations."""

    allowed_pairs = allowed_binding_pairs(available_tables, table_schemas)
    requested_items = [
        (
            str(item.get("binding_kind") or "unknown"),
            str(item.get("requested_term") or ""),
            bool(item.get("essential")),
        )
        for item in requested_terms
    ]
    requested_lookup = {
        (binding_kind, requested_term): essential
        for binding_kind, requested_term, essential in requested_items
    }
    bindings: list[SchemaGroundingBinding] = []
    unresolved_terms = list(grounding.unresolved_terms)
    warnings = list(grounding.warnings)
    invalid_bindings: list[str] = []
    seen_bindings: set[tuple[str, str, str, str]] = set()

    unresolved_keys = {(item.binding_kind, item.requested_term) for item in unresolved_terms}

    for binding in grounding.bindings:
        pair = (binding.table_name, binding.column_name)
        if pair not in allowed_pairs:
            invalid_bindings.append(f"{binding.table_name}.{binding.column_name}")
            key = (binding.binding_kind, binding.requested_term)
            if key not in unresolved_keys:
                unresolved_terms.append(
                    UnresolvedSchemaTerm(
                        requested_term=binding.requested_term,
                        binding_kind=binding.binding_kind,
                        reason=(
                            "binding referenced a table or column outside "
                            "the exact selected-table metadata"
                        ),
                        essential=requested_lookup.get(key, False),
                    )
                )
                unresolved_keys.add(key)
            continue

        dedupe_key = (
            binding.binding_kind,
            binding.requested_term,
            binding.table_name,
            binding.column_name,
        )
        if dedupe_key in seen_bindings:
            continue
        seen_bindings.add(dedupe_key)
        bindings.append(binding)

    if invalid_bindings:
        warnings.append(
            "Dropped invented schema bindings: " + ", ".join(sorted(set(invalid_bindings)))
        )

    seen_unresolved: set[tuple[str, str]] = set()
    deduped_unresolved: list[UnresolvedSchemaTerm] = []
    for item in unresolved_terms:
        key = (item.binding_kind, item.requested_term)
        if key in seen_unresolved:
            continue
        seen_unresolved.add(key)
        deduped_unresolved.append(
            item.model_copy(
                update={
                    "essential": item.essential or requested_lookup.get(key, False),
                }
            )
        )

    accounted_terms = {(binding.binding_kind, binding.requested_term) for binding in bindings} | {
        (item.binding_kind, item.requested_term) for item in deduped_unresolved
    }
    for binding_kind, requested_term, essential in requested_items:
        key = (binding_kind, requested_term)
        if key in accounted_terms:
            continue
        deduped_unresolved.append(
            UnresolvedSchemaTerm(
                requested_term=requested_term,
                binding_kind=binding_kind,
                reason="model did not account for requested term",
                essential=essential,
            )
        )
        accounted_terms.add(key)

    sanitized = SchemaGrounding(
        bindings=bindings,
        unresolved_terms=deduped_unresolved,
        warnings=warnings,
    )
    diagnostics = {
        "allowed_binding_count": len(allowed_pairs),
        "binding_count": len(bindings),
        "unresolved_count": len(deduped_unresolved),
        "invalid_bindings": invalid_bindings,
        "warning_count": len(warnings),
    }
    return sanitized, diagnostics


def allowed_binding_pairs(
    available_tables: Sequence[str],
    table_schemas: dict[str, TableSchema],
) -> set[tuple[str, str]]:
    """Return the finite exact binding universe for the current selected tables."""

    pairs: set[tuple[str, str]] = set()
    for table_name in available_tables:
        table = table_schemas.get(table_name)
        if table is None:
            continue
        for column in table.columns:
            pairs.add((table_name, column.name))
    return pairs


def render_grounding_block(grounding: SchemaGrounding | None) -> str | None:
    """Render grounded bindings and unresolved terms for SQL-stage prompts."""

    if grounding is None:
        return None
    if not grounding.bindings and not grounding.unresolved_terms and not grounding.warnings:
        return None

    lines = ["Grounded schema bindings:"]
    if grounding.bindings:
        for binding in grounding.bindings:
            lines.append(
                f"- {binding.binding_kind}: {binding.requested_term} -> "
                f"{binding.table_name}.{binding.column_name}"
            )
    else:
        lines.append("- none")

    if grounding.unresolved_terms:
        lines.append("Unresolved schema terms:")
        for item in grounding.unresolved_terms:
            essential = " essential" if item.essential else ""
            lines.append(f"- {item.binding_kind}{essential}: {item.requested_term} ({item.reason})")
        lines.append("Do not invent tables or columns to satisfy unresolved essential terms.")

    if grounding.warnings:
        lines.append("Grounding warnings:")
        for warning in grounding.warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines)
