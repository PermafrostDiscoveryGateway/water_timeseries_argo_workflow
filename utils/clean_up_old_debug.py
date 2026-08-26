import sys
import shutil
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os
from loguru import logger

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def find_old_debug_dirs(debug_dir: Path, num_days: float, now: float = None):
    """
    Find debug_* directories under debug_dir whose last-modified time is
    older than num_days.

    Args:
        debug_dir: Directory that contains the debug_* directories
        num_days: Age threshold in days
        now: Reference time (epoch seconds), defaults to time.time()

    Returns:
        list[Path]: debug directories older than the threshold
    """
    if now is None:
        now = time.time()

    max_age_seconds = num_days * 86400
    old_dirs = []

    for entry in sorted(debug_dir.glob('debug_*')):
        if not entry.is_dir():
            continue
        age_seconds = now - entry.stat().st_mtime
        if age_seconds > max_age_seconds:
            old_dirs.append(entry)

    return old_dirs


def main():
    logger.debug("Cleaning up old debug directories")
    env_path = None
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    debug_dir = os.environ.get('debug_dir')
    num_days = os.environ.get('num_days')
    dry_run = os.environ.get('DRY_RUN', 'False').lower() in ('true', '1', 'yes')

    if not debug_dir:
        logger.error("debug_dir environment variable not set")
        return

    if not num_days:
        logger.error("num_days environment variable not set")
        return

    num_days = float(num_days)
    debug_dir = Path(debug_dir)

    if not debug_dir.exists():
        logger.error(f"debug_dir {debug_dir} does not exist")
        return

    old_dirs = find_old_debug_dirs(debug_dir, num_days)

    if not old_dirs:
        logger.info(f"No debug directories older than {num_days} days found in {debug_dir}")
        return

    logger.info(f"Found {len(old_dirs)} debug directories older than {num_days} days in {debug_dir}")

    for old_dir in old_dirs:
        age_days = (time.time() - old_dir.stat().st_mtime) / 86400
        modified = datetime.fromtimestamp(old_dir.stat().st_mtime)
        if dry_run:
            logger.info(f"[DRY RUN] Would delete {old_dir} (last modified {modified}, {age_days:.1f} days old)")
        else:
            logger.info(f"Deleting {old_dir} (last modified {modified}, {age_days:.1f} days old)")
            shutil.rmtree(old_dir)


if __name__ == '__main__':
    main()
