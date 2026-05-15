"""Tests for curated large-schema summary metadata."""

from __future__ import annotations

import json

import pytest

from sol01.schema.large_schema_summaries import (
    FORBIDDEN_SUMMARY_TOKENS,
    load_large_schema_summary_registry,
)


def test_default_large_schema_summary_registry_loads_required_coverage():
    registry = load_large_schema_summary_registry()
    summary_ids = {summary.summary_id for summary in registry.summaries}

    assert {
        "github_repos_day_events",
        "bls_qcew_quarterly_area_industry",
        "covid19_usa_acs_county_fips",
        "census_bureau_acs_geography_year_estimates",
        "covid19_usa_vaccination_access_facility_boundary",
        "covid19_usafacts_wide_daily_counts",
        "covid19_jhu_csse_wide_daily_counts",
        "covid19_open_data_wide_public_health",
        "idc_dicom_imaging_metadata",
        "tcga_pancancer_clinical",
        "nppes_provider_registry",
        "usfs_fia_forest_inventory",
        "covid19_symptom_search_wide_symptoms",
    }.issubset(summary_ids)

    for summary in registry.summaries:
        assert summary.purpose
        assert summary.grain
        assert summary.stable_columns
        assert summary.repeated_column_rules
        assert summary.quote_spelling_rules
        assert 3 <= len(summary.examples) <= 5
        assert summary.aliases


def test_registry_matches_regex_families_exact_tables_and_duplicate_schema_copies():
    registry = load_large_schema_summary_registry()

    assert _ids(registry.match_table_ref("GITHUB_REPOS_DATE.DAY._20110212")) == [
        "github_repos_day_events"
    ]
    assert _ids(registry.match_table_ref("GITHUB_REPOS_DATE.DAY._20241022")) == [
        "github_repos_day_events"
    ]
    assert registry.match_table_ref("GITHUB_REPOS_DATE.DAY._20250101") == []

    assert _ids(registry.match_table_ref("BLS.BLS_QCEW._1990_Q1")) == [
        "bls_qcew_quarterly_area_industry"
    ]
    assert _ids(registry.match_table_ref("GOOGLE_DEI.BLS_QCEW._2019_Q2")) == [
        "bls_qcew_quarterly_area_industry"
    ]
    assert registry.match_table_ref("BLS.BLS_QCEW._2019_Q3") == []

    assert _ids(registry.match_table_ref("FEC.CENSUS_BUREAU_ACS.COUNTY_2021_1YR")) == [
        "census_bureau_acs_geography_year_estimates"
    ]
    assert _ids(registry.match_table_ref("COVID19_USA.CENSUS_BUREAU_ACS.COUNTY_2018_5YR")) == [
        "covid19_usa_acs_county_fips",
        "census_bureau_acs_geography_year_estimates",
    ]
    assert _ids(registry.match_table_ref("SDOH.CENSUS_BUREAU_ACS.BLOCKGROUP_2018_5YR")) == [
        "census_bureau_acs_geography_year_estimates"
    ]
    assert _ids(
        registry.match_table_ref(
            "COVID19_USA.COVID19_VACCINATION_ACCESS.FACILITY_BOUNDARY_US_ALL"
        )
    ) == ["covid19_usa_vaccination_access_facility_boundary"]

    assert _ids(
        registry.match_table_ref("COVID19_OPEN_WORLD_BANK.COVID19_OPEN_DATA.COMPATIBILITY_VIEW")
    ) == ["covid19_open_data_wide_public_health"]
    assert _ids(registry.match_table_ref("IDC.IDC_V17.DICOM_ALL")) == ["idc_dicom_imaging_metadata"]
    assert _ids(registry.match_table_ref("IDC_V17.DICOM_METADATA")) == [
        "idc_dicom_imaging_metadata"
    ]
    assert _ids(
        registry.match_table_ref(
            "COVID19_SYMPTOM_SEARCH.COVID19_SYMPTOM_SEARCH.SYMPTOM_SEARCH_SUB_REGION_2_WEEKLY"
        )
    ) == ["covid19_symptom_search_wide_symptoms"]


def test_summary_data_excludes_question_ids_gold_sql_and_answer_hints():
    registry = load_large_schema_summary_registry()
    payload = json.dumps(registry.model_dump(mode="json"), sort_keys=True).casefold()

    for forbidden_token in FORBIDDEN_SUMMARY_TOKENS:
        assert forbidden_token not in payload


def test_registry_loader_reports_clear_errors_for_malformed_records(tmp_path):
    malformed_path = tmp_path / "summaries.json"
    malformed_path.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "summary_id": "bad_summary",
                        "schema_copies": [
                            {"database": "DB", "schema_name": "PUBLIC"},
                        ],
                        "match": {"table_pattern": "["},
                        "purpose": "Bad malformed summary.",
                        "grain": "One row per example.",
                        "stable_columns": ["id"],
                        "repeated_column_rules": ["No repeated columns."],
                        "quote_spelling_rules": ["Use ID exactly."],
                        "examples": ["EXAMPLE", "EXAMPLE_2", "EXAMPLE_3"],
                        "aliases": ["bad"],
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid large schema summary registry") as exc_info:
        load_large_schema_summary_registry(malformed_path)

    assert "invalid table_pattern" in str(exc_info.value)


def _ids(summaries):
    return [summary.summary_id for summary in summaries]
