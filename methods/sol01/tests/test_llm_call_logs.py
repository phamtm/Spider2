"""Tests for reading per-instance LLM call logs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sol01.llm_call_logs import load_llm_call_log


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_load_llm_call_log_reads_valid_rows_in_order(tmp_path: Path):
    log_path = tmp_path / "local003.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "sequence": 1,
                "call_id": "0001-intent",
                "prompt_name": "intent",
                "status": "success",
                "started_at": "2026-05-02T10:00:00Z",
                "completed_at": "2026-05-02T10:00:01Z",
                "duration_ms": 1000,
                "model": "deepseek/deepseek-v4-pro",
                "request": {"user_prompt": "Question one."},
                "response": {"validated_output": {"summary": "ok"}},
                "attempts": [{"status": "success"}],
                "error": None,
                "prompt_sha256": "hash-1",
            },
            {
                "sequence": 2,
                "call_id": "0002-sql_generation",
                "prompt_name": "sql_generation",
                "status": "error",
                "started_at": "2026-05-02T10:01:00Z",
                "completed_at": "2026-05-02T10:01:03Z",
                "duration_ms": 3000,
                "model": "deepseek/deepseek-v4-pro",
                "request": {"user_prompt": "Question two."},
                "response": None,
                "attempts": [{"status": "error", "error": {"status_code": 400}}],
                "error": {"type": "ModelHTTPError", "status_code": 400},
                "prompt_sha256": "hash-2",
            },
            {
                "sequence": 3,
                "call_id": "0003-result_critic",
                "prompt_name": "result_critic",
                "status": "success",
                "started_at": "2026-05-02T10:02:00Z",
                "completed_at": "2026-05-02T10:02:05Z",
                "duration_ms": 5000,
                "model": "deepseek/deepseek-v4-pro",
                "request": {"user_prompt": "Question three."},
                "response": {"validated_output": {"confidence": 0.9}},
                "attempts": [
                    {"status": "error", "error": {"status_code": 429}},
                    {"status": "success"},
                ],
                "error": None,
                "prompt_sha256": "hash-3",
            },
        ],
    )

    result = load_llm_call_log(log_path)

    assert result.path == log_path
    assert result.errors == []
    assert [record.sequence for record in result.records] == [1, 2, 3]
    assert [record.prompt_name for record in result.records] == [
        "intent",
        "sql_generation",
        "result_critic",
    ]
    assert result.records[0].status == "success"
    assert result.records[0].started_at == datetime(2026, 5, 2, 10, 0, tzinfo=UTC)
    assert result.records[0].completed_at == datetime(2026, 5, 2, 10, 0, 1, tzinfo=UTC)
    assert result.records[1].error == {"type": "ModelHTTPError", "status_code": 400}
    assert result.records[2].attempts[0]["status"] == "error"
    assert result.records[2].attempts[1]["status"] == "success"
    assert result.records[2].duration_ms == 5000


def test_load_llm_call_log_reports_corrupted_rows_and_skips_them(tmp_path: Path):
    log_path = tmp_path / "local004.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sequence": 1,
                        "prompt_name": "intent",
                        "status": "success",
                        "model": "deepseek/deepseek-v4-pro",
                        "attempts": [],
                    }
                ),
                '{"sequence": 2, "prompt_name": "sql_generation"',
                json.dumps(
                    {
                        "sequence": 3,
                        "prompt_name": "result_critic",
                        "status": "error",
                        "model": "deepseek/deepseek-v4-pro",
                        "attempts": [{"status": "error"}],
                        "error": {"type": "ModelHTTPError"},
                    }
                ),
                "[]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = load_llm_call_log(log_path)

    assert [record.sequence for record in result.records] == [1, 3]
    assert result.records[0].prompt_name == "intent"
    assert result.records[1].status == "error"
    assert len(result.errors) == 2
    assert result.errors[0].line_number == 2
    assert result.errors[1].line_number == 4
    assert result.errors[0].path == log_path


def test_load_llm_call_log_missing_file_returns_empty_result(tmp_path: Path):
    log_path = tmp_path / "missing.jsonl"

    result = load_llm_call_log(log_path)

    assert result.path == log_path
    assert result.records == []
    assert result.errors == []
