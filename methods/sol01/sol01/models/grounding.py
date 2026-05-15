"""Grounded schema bindings used after table resolution."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SchemaBindingKind = Literal["metric", "filter", "native_term", "entity", "order", "unknown"]


class SchemaGroundingBinding(BaseModel):
    """One requested term bound to one exact selected-table column."""

    requested_term: str
    binding_kind: SchemaBindingKind = "unknown"
    table_name: str
    column_name: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    note: str | None = None


class UnresolvedSchemaTerm(BaseModel):
    """One requested term that could not be grounded to the selected tables."""

    requested_term: str
    binding_kind: SchemaBindingKind = "unknown"
    reason: str
    essential: bool = False


class SchemaGrounding(BaseModel):
    """Validated grounded schema bindings plus unresolved requested terms."""

    bindings: list[SchemaGroundingBinding] = Field(default_factory=list)
    unresolved_terms: list[UnresolvedSchemaTerm] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
