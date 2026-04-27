"""Tests for retrieval modes and the retrieval comparison experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sol01.analysis import compare_retrieval_modes
from sol01.config import RuntimeConfig
from sol01.llm import PromptSpec
from sol01.models import FinalAnswer, SchemaSelection, TableSelectionDecision, Task
from sol01.retrieval import retrieve_schema

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "retrieval_cases.json"


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


def test_retrieval_fixture_contains_expected_vague_cases():
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert len(payload["cases"]) >= 5
    assert {case["instance_id"] for case in payload["cases"]} >= {
        "local003",
        "local004",
        "local007",
        "local015",
        "local020",
    }


def test_retrieve_schema_defaults_to_lexical_mode():
    selection = retrieve_schema(
        "Which customers placed the highest value orders?",
        "E_commerce",
    )

    assert isinstance(selection, SchemaSelection)
    assert selection.retrieval_mode == "lexical"
    assert selection.selection_prompt_chars == 0
    assert selection.candidate_table_count == 11


def test_retrieve_schema_llm_only_uses_schema_selector_and_filters_unknown_tables():
    llm = FakeSelectorLLM(
        TableSelectionDecision(
            selected_tables=["orders", "missing_table", "customers", "orders"],
            rationale="Orders and customers are the core business entities here.",
            confidence=0.82,
        )
    )

    selection = retrieve_schema(
        "Which customers placed the highest value orders?",
        "E_commerce",
        retrieval_mode="llm_only",
        llm_client=llm,
        max_tables=3,
        max_expanded_tables=4,
    )

    assert selection.retrieval_mode == "llm_only"
    assert selection.selected_tables == ["orders", "customers"]
    assert selection.selection_prompt_chars > 0
    assert selection.candidate_table_count == 11
    assert llm.calls[0]["prompt_name"] == "schema_selection"
    assert "Schema summary:" in llm.calls[0]["user_prompt"]
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
        "E_commerce",
        retrieval_mode="llm_only",
        llm_client=llm,
    )

    assert selection.selected_tables == []
    assert selection.expanded_tables == []
    assert selection.confidence == 0.0
    assert "No valid tables matched the schema summary." in selection.rationale


def test_compare_retrieval_modes_writes_comparison_artifacts(
    monkeypatch,
    tmp_path: Path,
):
    fixture_path = tmp_path / "retrieval_cases.json"
    fixture_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "instance_id": "local003",
                        "db": "E_commerce",
                        "question": "RFM question",
                        "expected_tables": ["customers", "orders"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_run_task(task: Task, *, run_paths, config, force=False, **_: Any) -> FinalAnswer:
        selection = {
            "db": task.db,
            "retrieval_mode": config.retrieval_mode,
            "selected_tables": ["orders"]
            if config.retrieval_mode == "lexical"
            else ["orders", "customers"],
            "expanded_tables": ["orders"]
            if config.retrieval_mode == "lexical"
            else ["orders", "customers"],
            "rationale": f"{config.retrieval_mode} selection",
            "confidence": 0.8,
            "selection_prompt_chars": 0 if config.retrieval_mode == "lexical" else 240,
            "candidate_table_count": 11,
        }
        trace = {
            "instance_id": task.instance_id,
            "db": task.db,
            "question": task.question,
            "status": "failed" if config.retrieval_mode == "lexical" else "success",
            "schema_selection": selection,
            "sql_path": str(run_paths.sql_dir / f"{task.instance_id}.sql"),
        }
        (run_paths.sql_dir / f"{task.instance_id}.sql").write_text("SELECT 1;\n", encoding="utf-8")
        (run_paths.traces_dir / f"{task.instance_id}.json").write_text(
            json.dumps(trace, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return FinalAnswer(
            instance_id=task.instance_id,
            status=trace["status"],
            sql="SELECT 1;",
            csv_path=None,
            trace_path=str(run_paths.traces_dir / f"{task.instance_id}.json"),
        )

    monkeypatch.setattr("sol01.analysis.run_task", fake_run_task)

    report = compare_retrieval_modes(
        "retrieval-exp",
        config=RuntimeConfig(api_key="test-key"),
        fixture_path=fixture_path,
        outputs_root=tmp_path / "outputs",
    )

    compare_json = tmp_path / "outputs" / "retrieval-exp" / "analysis" / "retrieval_compare.json"
    compare_md = tmp_path / "outputs" / "retrieval-exp" / "analysis" / "retrieval_compare.md"
    payload = json.loads(compare_json.read_text(encoding="utf-8"))

    assert report["summary"]["lexical"]["miss_count"] == 1
    assert report["summary"]["llm_only"]["miss_count"] == 0
    assert compare_json.exists()
    assert compare_md.exists()
    assert payload["cases"][0]["modes"]["llm_only"]["final_sql_outcome"] == "success"
    assert "llm_only" in compare_md.read_text(encoding="utf-8")
