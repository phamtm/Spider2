"""Tests for deterministic SQL identifier auto-quoting."""

from sol01.execution.normalization import auto_quote_identifiers
from sol01.models import ColumnSchema, TableSchema


def _make_schemas(columns: dict[str, list[str]]) -> dict[str, TableSchema]:
    """Build table schemas from ``{table_name: [col_names]}``."""
    schemas: dict[str, TableSchema] = {}
    for table_name, col_names in columns.items():
        schemas[table_name] = TableSchema(
            name=table_name,
            ddl="",
            columns=[ColumnSchema(name=c, type="TEXT") for c in col_names],
            searchable_text="",
        )
    return schemas


BIKESHARE_SCHEMAS = _make_schemas(
    {
        "AUSTIN.AUSTIN.BIKESHARE_STATIONS": [
            "station_id",
            "status",
            "modified_date",
        ],
        "AUSTIN.AUSTIN.BIKESHARE_TRIPS": [
            "trip_id",
            "station_id",
        ],
    }
)

DICOM_SCHEMAS = _make_schemas(
    {
        "IDC.IDC_V17.DICOM_PIVOT": [
            "StudyInstanceUID",
            "SegmentedPropertyTypeCodeSequence",
            "collection_id",
        ],
    }
)

ALL_UPPER_SCHEMAS = _make_schemas(
    {
        "DB.SCHEMA.TABLE": [
            "ID",
            "NAME",
            "CREATED_AT",
        ],
    }
)


class TestAutoQuoteBasic:
    def test_unquoted_lowercase_column_gets_quoted(self):
        sql = "SELECT station_id FROM AUSTIN.AUSTIN.BIKESHARE_STATIONS"
        result = auto_quote_identifiers(sql, BIKESHARE_SCHEMAS)
        assert '"station_id"' in result
        assert result.count("station_id") == 1

    def test_unquoted_mixed_case_column_gets_quoted(self):
        sql = "SELECT StudyInstanceUID FROM IDC.IDC_V17.DICOM_PIVOT"
        result = auto_quote_identifiers(sql, DICOM_SCHEMAS)
        assert '"StudyInstanceUID"' in result

    def test_already_quoted_column_unchanged(self):
        sql = 'SELECT "collection_id" FROM IDC.IDC_V17.DICOM_PIVOT'
        result = auto_quote_identifiers(sql, DICOM_SCHEMAS)
        assert '"collection_id"' in result

    def test_already_upper_column_unchanged(self):
        sql = "SELECT ID FROM DB.SCHEMA.TABLE"
        result = auto_quote_identifiers(sql, ALL_UPPER_SCHEMAS)
        assert '"ID"' not in result


class TestAutoQuoteQualified:
    def test_qualified_column_gets_quoted(self):
        sql = "SELECT b.station_id FROM AUSTIN.AUSTIN.BIKESHARE_STATIONS AS b"
        result = auto_quote_identifiers(sql, BIKESHARE_SCHEMAS)
        assert 'b."station_id"' in result or '"station_id"' in result

    def test_where_clause_column_gets_quoted(self):
        sql = "SELECT station_id FROM AUSTIN.AUSTIN.BIKESHARE_STATIONS WHERE status = 'active'"
        result = auto_quote_identifiers(sql, BIKESHARE_SCHEMAS)
        assert '"station_id"' in result
        assert '"status"' in result


class TestAutoQuoteMultipleColumns:
    def test_multiple_unquoted_columns_all_fixed(self):
        sql = "SELECT station_id, status FROM AUSTIN.AUSTIN.BIKESHARE_STATIONS"
        result = auto_quote_identifiers(sql, BIKESHARE_SCHEMAS)
        assert '"station_id"' in result
        assert '"status"' in result


class TestAutoQuoteEdgeCases:
    def test_empty_schemas_returns_original(self):
        sql = "SELECT station_id FROM some_table"
        result = auto_quote_identifiers(sql, {})
        assert result == sql

    def test_unparseable_sql_returns_original(self):
        sql = "SELECT FROM WHERE"
        result = auto_quote_identifiers(sql, BIKESHARE_SCHEMAS)
        assert result == sql

    def test_cte_column_gets_quoted(self):
        sql = (
            "WITH active AS (SELECT station_id FROM AUSTIN.AUSTIN.BIKESHARE_STATIONS) "
            "SELECT station_id FROM active"
        )
        result = auto_quote_identifiers(sql, BIKESHARE_SCHEMAS)
        assert '"station_id"' in result

    def test_group_by_column_gets_quoted(self):
        sql = "SELECT status, COUNT(*) FROM AUSTIN.AUSTIN.BIKESHARE_STATIONS GROUP BY status"
        result = auto_quote_identifiers(sql, BIKESHARE_SCHEMAS)
        assert '"status"' in result

    def test_order_by_column_gets_quoted(self):
        sql = "SELECT station_id FROM AUSTIN.AUSTIN.BIKESHARE_STATIONS ORDER BY modified_date"
        result = auto_quote_identifiers(sql, BIKESHARE_SCHEMAS)
        assert '"modified_date"' in result


class TestAutoQuoteCrossTableCollision:
    """Columns with same lowercase name but different cases on different tables."""

    COLLISION_SCHEMAS = _make_schemas(
        {
            # SALESPERSON has lowercase businessentityid
            "DB.SCHEMA.SALESPERSON": ["businessentityid", "name"],
            # SALESPERSONQUOTAHISTORY has mixed-case BusinessEntityID and SalesQuota
            "DB.SCHEMA.SALESPERSONQUOTAHISTORY": ["BusinessEntityID", "SalesQuota", "QuotaDate"],
        }
    )

    def test_qualified_cols_disambiguated_by_table_alias(self):
        sql = (
            "SELECT sp.businessentityid, sqh.SalesQuota "
            "FROM DB.SCHEMA.SALESPERSON AS sp "
            "JOIN DB.SCHEMA.SALESPERSONQUOTAHISTORY AS sqh "
            "ON sqh.BusinessEntityID = sp.businessentityid"
        )
        result = auto_quote_identifiers(sql, self.COLLISION_SCHEMAS)
        assert 'sp."businessentityid"' in result
        assert 'sqh."SalesQuota"' in result
        assert 'sqh."BusinessEntityID"' in result

    def test_unqualified_col_in_single_source_subquery(self):
        sql = (
            "SELECT sp.businessentityid "
            "FROM DB.SCHEMA.SALESPERSON AS sp "
            "JOIN (SELECT BusinessEntityID, SalesQuota "
            "FROM DB.SCHEMA.SALESPERSONQUOTAHISTORY) AS sqh "
            "ON sqh.BusinessEntityID = sp.businessentityid"
        )
        result = auto_quote_identifiers(sql, self.COLLISION_SCHEMAS)
        assert '"BusinessEntityID"' in result
        assert '"SalesQuota"' in result


class TestAutoQuoteCteQualifiedReference:
    """Qualified references to CTE output columns should be quoted via global fallback."""

    def test_cte_qualified_uppercase_reference_gets_quoted(self):
        # CTE outputs station_id (lowercase, from inner quoted column).
        # Outer S.STATION_ID is unquoted uppercase — must be fixed to S."station_id".
        sql = (
            'WITH all_st AS (SELECT "station_id" FROM AUSTIN.AUSTIN.BIKESHARE_STATIONS) '
            "SELECT S.STATION_ID FROM all_st AS S"
        )
        result = auto_quote_identifiers(sql, BIKESHARE_SCHEMAS)
        assert 'S."station_id"' in result


class TestAutoQuoteSelectAliasNotDoubleQuoted:
    """SELECT aliases used in GROUP BY / ORDER BY must not be falsely quoted."""

    def test_select_alias_in_group_by_not_quoted_as_column(self):
        sql = (
            "SELECT DATE_PART(YEAR, modified_date) AS yr, status, COUNT(*) "
            "FROM AUSTIN.AUSTIN.BIKESHARE_STATIONS "
            "GROUP BY yr, status"
        )
        result = auto_quote_identifiers(sql, BIKESHARE_SCHEMAS)
        # status and modified_date should be quoted; yr is an alias, not a column
        assert '"status"' in result
        assert '"modified_date"' in result
        assert '"yr"' not in result


class TestAutoQuoteNoOpForCorrect:
    def test_no_change_when_all_upper(self):
        sql = "SELECT ID, NAME FROM DB.SCHEMA.TABLE"
        result = auto_quote_identifiers(sql, ALL_UPPER_SCHEMAS)
        assert result == sql

    def test_no_change_when_already_quoted(self):
        sql = 'SELECT "station_id", "status" FROM AUSTIN.AUSTIN.BIKESHARE_STATIONS'
        result = auto_quote_identifiers(sql, BIKESHARE_SCHEMAS)
        assert result == sql
