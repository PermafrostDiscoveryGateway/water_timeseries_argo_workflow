# Shared summer-season / test-run date gating for the near-real-time scripts.
#
# Normal runs only fire in June-September, and only a few days into the month
# (so the prior month's upstream data is actually available) - see the
# `TODAY_DAY > N` checks next to each call site. A test run needs to be able
# to fire on any day, against whichever month is actually testable. The
# current month is never a candidate - even in-season it's still in
# progress and won't have data yet - so this always targets the most recent
# *complete* summer month, starting the search at last month.
#
# A test run can also be pinned to one specific month by setting
# `target_date=YYYY-MM` in the .env (used by the local snakemake test
# pipeline), instead of whatever month "today" happens to resolve to.
import os
import re
from datetime import datetime

SUMMER_MONTHS = [6, 7, 8, 9]


def is_test_run() -> bool:
    return os.environ.get("test_run", "False").lower() in ("true", "1", "yes")


def target_date_override() -> "datetime | None":
    """Explicit `target_date=YYYY-MM` from the env, or None if unset."""
    value = os.environ.get("target_date", "").strip()
    if not value:
        return None
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
        raise ValueError(f"target_date must be YYYY-MM, got {value!r}")
    return datetime.strptime(value, "%Y-%m")


def should_run_last_attempts(today: datetime = None, min_day: int = 15) -> bool:
    """Gate for download_region_last_attempts.py.

    Dynamic World ingestion for a just-finished month can lag by a couple of
    weeks, so a lake confirmed "no data" early in the month (day 3-14, when
    the regular download/backfill scripts run) may just not be ingested yet.
    This gate holds the last-attempts script back until `min_day`, giving
    ingestion time to catch up before treating a lake as permanently no-data.
    """
    if is_test_run():
        return True
    if today is None:
        today = datetime.now()
    return today.day >= min_day and today.month - 1 in SUMMER_MONTHS


def most_recent_summer_month(today: datetime = None) -> datetime:
    override = target_date_override()
    if override is not None:
        return override
    if today is None:
        today = datetime.now()
    year, month = today.year, today.month
    month -= 1
    if month == 0:
        month = 12
        year -= 1
    while month not in SUMMER_MONTHS:
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return datetime(year, month, 1)
