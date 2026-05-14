"""Question-derived constraint helpers for table-family resolution."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date

from sol01.models import SchemaPlanningConstraints

_BROAD_RANGE_RE = re.compile(
    r"\b(all|any|each|every|history|historical|daily|monthly|weekly|yearly|annual|"
    r"range|between|from|through|across|over time|time series)\b",
    re.I,
)
_DATE_RE = re.compile(r"\b(?P<year>(?:19|20)\d{2})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_VERSION_RE = re.compile(r"\bv(?:ersion)?[_ -]?(?P<value>\d+)\b", re.I)


def merge_question_constraints(
    constraints: SchemaPlanningConstraints | None,
    question: str,
) -> SchemaPlanningConstraints:
    """Add clear dates, years, and version mentions from the user question."""

    base = constraints or SchemaPlanningConstraints()
    question_years = [int(match.group(0)) for match in _YEAR_RE.finditer(question)]
    years = stable_unique_int([*base.years, *question_years])
    dates = [parse_date(match.group(0)) for match in _DATE_RE.finditer(question)]
    dates = [item for item in dates if item is not None]
    date_start = base.date_start
    date_end = base.date_end
    if dates:
        date_start = date_start or min(dates).isoformat()
        date_end = date_end or max(dates).isoformat()
    elif len(question_years) >= 2:
        first_year = min(question_years)
        last_year = max(question_years)
        date_start = date_start or date(first_year, 1, 1).isoformat()
        date_end = date_end or date(last_year, 12, 31).isoformat()
    version = base.version
    if version is None and (match := _VERSION_RE.search(question)):
        version = match.group("value")
    return base.model_copy(
        update={"years": years, "date_start": date_start, "date_end": date_end, "version": version}
    )


def has_explicit_constraints(constraints: SchemaPlanningConstraints) -> bool:
    """Return True when family member selection has explicit narrowing hints."""

    return bool(
        constraints.date_start
        or constraints.date_end
        or constraints.years
        or constraints.suffixes
        or constraints.version is not None
    )


def question_asks_broad_range(question: str) -> bool:
    """Return True when question wording implies a broad historical scan."""

    if _BROAD_RANGE_RE.search(question):
        return True
    return len(_DATE_RE.findall(question)) >= 2 or len(_YEAR_RE.findall(question)) >= 2


def member_matches_years(
    value: str,
    raw_value: str,
    member_date: date | None,
    years: list[int],
) -> bool:
    """Return True when a member matches one of the requested years."""

    year_set = set(years)
    if member_date is not None and member_date.year in year_set:
        return True
    return any(str(year) in {value, raw_value} for year in year_set)


def dimension_date(value: str) -> date | None:
    """Interpret a suffix value as a day or year when possible."""

    if re.fullmatch(r"(?:19|20)\d{2}", value):
        return date(int(value), 1, 1)
    return parse_date(value)


def parse_date(value: str | None) -> date | None:
    """Parse one date-like suffix value."""

    if not value:
        return None
    match = _DATE_RE.search(str(value))
    if not match:
        return None
    try:
        return date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    except ValueError:
        return None


def normalize_constraint(value: object) -> str:
    """Normalize free-form suffix-like values before comparison."""

    return str(value).strip().strip("_-").lower()


def normalize_version(value: object) -> str:
    """Normalize version strings to their numeric component."""

    return re.sub(r"[^0-9]", "", str(value))


def stable_unique_int(values: Iterable[int]) -> list[int]:
    """Return stable sorted integer values with duplicates removed."""

    return sorted({value for value in values})
