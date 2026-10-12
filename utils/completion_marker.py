import os
from pathlib import Path
from typing import Iterable, Optional

from loguru import logger

MARKER_SUFFIX = ".complete"


def marker_path(nc_file) -> Path:
    """Marker file that sits next to nc_file once it has been fully written."""
    return Path(str(nc_file) + MARKER_SUFFIX)


def clear_completion_marker(nc_file) -> None:
    """Remove the marker before (re)writing nc_file so readers skip it while it is in flux."""
    marker_path(nc_file).unlink(missing_ok=True)


def mark_complete(nc_file) -> None:
    """Write the marker after nc_file has been fully written and closed."""
    marker_path(nc_file).touch()


def find_most_recent_completed(nc_files: Iterable) -> Optional[str]:
    """
    Return the newest (by mtime) file that has a completion marker.

    Files written before markers were introduced have none, so if no file has a
    marker we fall back to the newest file by mtime and log a warning.
    """
    nc_files = list(nc_files)
    if not nc_files:
        return None

    completed = [f for f in nc_files if marker_path(f).exists()]
    if completed:
        return max(completed, key=os.path.getmtime)

    logger.warning("No .nc file has a completion marker; falling back to newest by mtime")
    return max(nc_files, key=os.path.getmtime)
