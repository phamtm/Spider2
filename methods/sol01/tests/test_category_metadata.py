"""Tests for Spider2-Snow category batch metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sol01.category_metadata import (
    CategoryMetadataValidationError,
    load_category_metadata,
    load_category_metadata_map,
    write_category_metadata,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_load_category_metadata_uses_dataset_order(tmp_path: Path):
    dataset = tmp_path / "spider2-snow.jsonl"
    _write_jsonl(
        dataset,
        [
            {"instance_id": "sf_a", "instruction": "q", "db_id": "DB", "external_knowledge": None},
            {"instance_id": "sf_b", "instruction": "q", "db_id": "DB", "external_knowledge": None},
        ],
    )
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    _write_jsonl(
        batch_dir / "batch_01.jsonl",
        [
            {"instance_id": "sf_b", "primary_tier": 2, "tags": ["alpha"]},
            {"instance_id": "sf_a", "primary_tier": 1, "tags": ["alpha"]},
        ],
    )

    records = load_category_metadata(
        dataset_path=dataset,
        batch_dir=batch_dir,
        allowed_tags={"alpha"},
    )

    assert [record.instance_id for record in records] == ["sf_a", "sf_b"]
    assert load_category_metadata_map(
        dataset_path=dataset,
        batch_dir=batch_dir,
        allowed_tags={"alpha"},
    ) == {record.instance_id: record for record in records}


def test_load_category_metadata_rejects_duplicates(tmp_path: Path):
    dataset = tmp_path / "spider2-snow.jsonl"
    _write_jsonl(
        dataset,
        [
            {"instance_id": "sf_a", "instruction": "q", "db_id": "DB", "external_knowledge": None},
        ],
    )
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    _write_jsonl(
        batch_dir / "batch_01.jsonl",
        [{"instance_id": "sf_a", "primary_tier": 1, "tags": ["alpha"]}],
    )
    _write_jsonl(
        batch_dir / "batch_02.jsonl",
        [{"instance_id": "sf_a", "primary_tier": 1, "tags": ["alpha"]}],
    )

    with pytest.raises(CategoryMetadataValidationError, match="duplicate metadata row"):
        load_category_metadata(dataset_path=dataset, batch_dir=batch_dir, allowed_tags={"alpha"})


def test_load_category_metadata_rejects_invalid_tier(tmp_path: Path):
    dataset = tmp_path / "spider2-snow.jsonl"
    _write_jsonl(
        dataset,
        [
            {"instance_id": "sf_a", "instruction": "q", "db_id": "DB", "external_knowledge": None},
        ],
    )
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    _write_jsonl(
        batch_dir / "batch_01.jsonl",
        [{"instance_id": "sf_a", "primary_tier": 13, "tags": ["alpha"]}],
    )

    with pytest.raises(CategoryMetadataValidationError, match="invalid primary_tier"):
        load_category_metadata(dataset_path=dataset, batch_dir=batch_dir, allowed_tags={"alpha"})


def test_load_category_metadata_rejects_unknown_tags(tmp_path: Path):
    dataset = tmp_path / "spider2-snow.jsonl"
    _write_jsonl(
        dataset,
        [
            {"instance_id": "sf_a", "instruction": "q", "db_id": "DB", "external_knowledge": None},
        ],
    )
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    _write_jsonl(
        batch_dir / "batch_01.jsonl",
        [{"instance_id": "sf_a", "primary_tier": 1, "tags": ["unknown_tag"]}],
    )

    with pytest.raises(CategoryMetadataValidationError, match="unknown tag unknown_tag"):
        load_category_metadata(dataset_path=dataset, batch_dir=batch_dir, allowed_tags={"alpha"})


def test_load_category_metadata_rejects_unknown_instance_ids(tmp_path: Path):
    dataset = tmp_path / "spider2-snow.jsonl"
    _write_jsonl(
        dataset,
        [
            {"instance_id": "sf_a", "instruction": "q", "db_id": "DB", "external_knowledge": None},
        ],
    )
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    _write_jsonl(
        batch_dir / "batch_01.jsonl",
        [{"instance_id": "sf_unknown", "primary_tier": 1, "tags": ["alpha"]}],
    )

    with pytest.raises(CategoryMetadataValidationError, match="unknown instance_id sf_unknown"):
        load_category_metadata(dataset_path=dataset, batch_dir=batch_dir, allowed_tags={"alpha"})


def test_load_category_metadata_reads_repo_batches():
    records = load_category_metadata()

    assert len(records) == 547
    assert records[0].instance_id == "sf_bq011"
    assert records[-1].instance_id == "sf014"


def test_write_category_metadata_exports_merged_file(tmp_path: Path):
    dataset = tmp_path / "spider2-snow.jsonl"
    _write_jsonl(
        dataset,
        [
            {"instance_id": "sf_a", "instruction": "q", "db_id": "DB", "external_knowledge": None},
            {"instance_id": "sf_b", "instruction": "q", "db_id": "DB", "external_knowledge": None},
        ],
    )
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    _write_jsonl(
        batch_dir / "batch_01.jsonl",
        [
            {"instance_id": "sf_b", "primary_tier": 2, "tags": ["alpha"], "difficulty_notes": None},
            {
                "instance_id": "sf_a",
                "primary_tier": 1,
                "tags": ["alpha"],
                "difficulty_notes": "note",
            },
        ],
    )
    output_path = tmp_path / "category_metadata.jsonl"

    written_path = write_category_metadata(
        output_path=output_path,
        dataset_path=dataset,
        batch_dir=batch_dir,
        allowed_tags={"alpha"},
    )

    assert written_path == output_path
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        '{"instance_id":"sf_a","primary_tier":1,"tags":["alpha"],"difficulty_notes":"note"}',
        '{"instance_id":"sf_b","primary_tier":2,"tags":["alpha"]}',
    ]
