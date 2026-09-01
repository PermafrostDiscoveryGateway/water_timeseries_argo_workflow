# Shared summer-season / test-run date gating for the near-real-time scripts.
#
# Normal runs only fire in June-September, and only a few days into the month
# (so the prior month's upstream data is actually available) - see the
# `TODAY_DAY > N` checks next to each call site. A test run needs to be able
# to fire on any day, against whichever month is actually testable, so it
# always targets the current month if that's a summer month, otherwise the
# most recent summer month before today.
import os
from datetime import datetime

SUMMER_MONTHS = [6, 7, 8, 9]


def is_test_run() -> bool:
    return os.environ.get("test_run", "False").lower() in ("true", "1", "yes")


def most_recent_summer_month(today: datetime = None) -> datetime:
    if today is None:
        today = datetime.now()
    year, month = today.year, today.month
    while month not in SUMMER_MONTHS:
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return datetime(year, month, 1)
