"""Tests for LLM-backed schema retrieval."""

from __future__ import annotations

from typing import Any

from sol01.llm.client import PromptSpec
from sol01.models import SchemaSelection, TableSelectionDecision
from sol01.schema.retrieval import retrieve_schema

CUSTOMERS = "E_COMMERCE.E_COMMERCE.CUSTOMERS"
ORDERS = "E_COMMERCE.E_COMMERCE.ORDERS"


class FakeSelectorLLM:
    """Small fake selector that returns one queued schema-selection decision."""

    def __init__(self, decision: TableSelectionDecision) -> None:
        self.decision = decision
        self.calls: list[dict[str, Any]] = []

    def load_prompt(self, prompt_name: str) -> PromptSpec:
        return PromptSpec(name=prompt_name, text="schema prompt", sha256="hash-schema")

    def run_structured_with_prompt(
        self,
        user_prompt: str,
        *,
        prompt: PromptSpec,
        output_type: type[Any],
        model: Any = None,
    ) -> Any:
        self.calls.append(
            {
                "prompt_name": prompt.name,
                "user_prompt": user_prompt,
                "output_type": output_type,
            }
        )
        assert output_type is TableSelectionDecision
        return self.decision


def test_retrieve_schema_defaults_to_llm_only_mode():
    llm = FakeSelectorLLM(
        TableSelectionDecision(
            selected_tables=["orders", "customers"],
            rationale="Orders and customers are the core business entities here.",
            confidence=0.82,
        )
    )

    selection = retrieve_schema(
        "Which customers placed the highest value orders?",
        "E_COMMERCE",
        llm_client=llm,
    )

    assert isinstance(selection, SchemaSelection)
    assert selection.retrieval_mode == "llm_only"
    assert selection.selection_prompt_chars > 0
    assert selection.candidate_table_count == 11
    assert selection.selected_tables == [ORDERS, CUSTOMERS]
    assert selection.expanded_tables == selection.selected_tables


def test_retrieve_schema_llm_only_uses_schema_selector_and_filters_unknown_tables():
    llm = FakeSelectorLLM(
        TableSelectionDecision(
            selected_tables=[
                "orders",
                "missing_table",
                "customers",
                "order_items",
                "products",
                "orders",
            ],
            rationale="Orders and customers are the core business entities here.",
            confidence=0.82,
        )
    )

    selection = retrieve_schema(
        "Which customers placed the highest value orders?",
        "E_COMMERCE",
        llm_client=llm,
    )

    assert selection.retrieval_mode == "llm_only"
    assert selection.selected_tables == [
        ORDERS,
        CUSTOMERS,
        "E_COMMERCE.E_COMMERCE.ORDER_ITEMS",
        "E_COMMERCE.E_COMMERCE.PRODUCTS",
    ]
    assert selection.expanded_tables == selection.selected_tables
    assert selection.selection_prompt_chars > 0
    assert selection.candidate_table_count == 11
    assert llm.calls[0]["prompt_name"] == "schema_selection"
    assert "Schema summary:" in llm.calls[0]["user_prompt"]
    assert "plausibly required" in llm.calls[0]["user_prompt"]
    assert "join and bridge tables" in llm.calls[0]["user_prompt"]
    assert "clearly irrelevant" in llm.calls[0]["user_prompt"]
    assert "clearly grounded" in llm.calls[0]["user_prompt"]
    assert "unambiguously match" in llm.calls[0]["user_prompt"]
    assert "column-name semantics" in llm.calls[0]["user_prompt"]
    assert "smallest" not in llm.calls[0]["user_prompt"].lower()
    assert "at most" not in llm.calls[0]["user_prompt"].lower()
    assert "missing_table" not in selection.selected_tables


def test_retrieve_schema_llm_only_surfaces_empty_valid_selection():
    llm = FakeSelectorLLM(
        TableSelectionDecision(
            selected_tables=["missing_table", "also_missing"],
            rationale="These names sounded right.",
            confidence=0.7,
        )
    )

    selection = retrieve_schema(
        "Which customers placed the highest value orders?",
        "E_COMMERCE",
        llm_client=llm,
    )

    assert selection.selected_tables == []
    assert selection.expanded_tables == []
    assert selection.confidence == 0.0
    assert "No valid tables matched the schema summary." in selection.rationale


def test_retrieve_schema_llm_only_accepts_suffix_matches():
    llm = FakeSelectorLLM(
        TableSelectionDecision(
            selected_tables=["E_COMMERCE.ORDERS", "E_COMMERCE.CUSTOMERS"],
            rationale="Orders and customers are the core business entities here.",
            confidence=0.82,
        )
    )

    selection = retrieve_schema(
        "Which customers placed the highest value orders?",
        "E_COMMERCE",
        llm_client=llm,
    )

    assert selection.selected_tables == [ORDERS, CUSTOMERS]
    assert selection.expanded_tables == selection.selected_tables
