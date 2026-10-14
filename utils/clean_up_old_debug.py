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


def newest_mtime(path: Path) -> float:
    """
    Return the newest mtime found by walking the tree under path.

    A directory's own mtime only changes when direct entries are added or
    removed, so files and nested directories are included to detect jobs that
    are still writing. Symlinks are not followed.
    """
    newest = path.lstat().st_mtime
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            try:
                newest = max(newest, os.lstat(os.path.join(root, name)).st_mtime)
            except OSError:
                # entry removed while walking
                continue
    return newest


def find_old_debug_dirs(debug_dir: Path, num_days: float, now: float = None):
    """
    Find debug_* directories under debug_dir whose newest modification time
    (across the whole tree) is older than num_days. Symlinks and directories
    that resolve outside debug_dir are skipped.

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
    resolved_debug_dir = debug_dir.resolve()

    for entry in sorted(debug_dir.glob('debug_*')):
        if entry.is_symlink() or not entry.is_dir():
            continue
        if not entry.resolve().is_relative_to(resolved_debug_dir):
            logger.warning(f"Skipping {entry}: resolves outside {debug_dir}")
            continue
        age_seconds = now - newest_mtime(entry)
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
        sys.exit(1)

    if not num_days:
        logger.error("num_days environment variable not set")
        sys.exit(1)

    try:
        num_days = float(num_days)
    except ValueError:
        logger.error(f"num_days must be a number, got {num_days!r}")
        sys.exit(1)
    debug_dir = Path(debug_dir)

    if not debug_dir.exists():
        logger.error(f"debug_dir {debug_dir} does not exist")
        sys.exit(1)

    old_dirs = find_old_debug_dirs(debug_dir, num_days)

    if not old_dirs:
        logger.info(f"No debug directories older than {num_days} days found in {debug_dir}")
        return

    logger.info(f"Found {len(old_dirs)} debug directories older than {num_days} days in {debug_dir}")

    failures = []
    for old_dir in old_dirs:
        try:
            newest = newest_mtime(old_dir)
            age_days = (time.time() - newest) / 86400
            modified = datetime.fromtimestamp(newest)
            if dry_run:
                logger.info(f"[DRY RUN] Would delete {old_dir} (last modified {modified}, {age_days:.1f} days old)")
            else:
                logger.info(f"Deleting {old_dir} (last modified {modified}, {age_days:.1f} days old)")
                shutil.rmtree(old_dir)
        except Exception as e:
            logger.error(f"Failed to delete {old_dir}: {e}")
            failures.append(old_dir)

    if dry_run:
        logger.info(f"[DRY RUN] Would delete {len(old_dirs) - len(failures)} debug directories")

    if failures:
        logger.error(f"Failed to delete {len(failures)} of {len(old_dirs)} debug directories: {[str(f) for f in failures]}")
        sys.exit(1)


if __name__ == '__main__':
    main()
