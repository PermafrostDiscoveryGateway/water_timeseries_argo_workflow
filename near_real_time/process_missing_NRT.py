"""Reprocess a single month of NRT data on demand.

Unlike process_NRT.py (which always targets the previous month), this script takes an
explicit target month (e.g. "2026-06") as a command-line argument, so it can be used to
backfill/fix a month that was missed or came out incomplete.

Usage:
    python process_missing_NRT.py 2026-06 [path/to/.env]
"""
from process_NRT import process_single_date_for_region
import sys
import re
import time
import glob
import os
from typing import Dict, Any
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

DATE_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def main():
    logger.debug("=" * 80)
    logger.debug("PROCESS_MISSING_NRT.PY STARTED")
    logger.debug("=" * 80)

    if len(sys.argv) < 2:
        logger.error('Usage: python process_missing_NRT.py <YYYY-MM> [path/to/.env]')
        sys.exit(1)

    target_date = sys.argv[1]
    if not DATE_PATTERN.match(target_date):
        logger.error(f"Invalid date '{target_date}' - expected format YYYY-MM, e.g. 2026-06")
        sys.exit(1)

    env_path = None
    if len(sys.argv) > 2:
        env_path = sys.argv[2]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    dynamic_world_data_dir = os.environ['dynamic_world_data']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
    if not all_dynamic_world_files:
        logger.error(f"No .nc files found in {dynamic_world_data_dir}")
        return {'success': False, 'error': 'No .nc files found'}

    region_name = os.environ.get("region_name", "ALL")
    id_chunk_size = int(os.environ.get("id_chunk_size", 500))
    save_interval = int(os.environ.get("save_interval", 1))
    n_jobs = int(os.environ.get("n_jobs", 1))

    if region_name == "ALL":
        regions_to_process = ['TEST', 'ALASKA', 'CANADA', 'EURASIA1', 'EURASIA2', 'EURASIA3']
        logger.info(f"Processing ALL main regions: {regions_to_process}")
    else:
        regions_to_process = [region_name]
        logger.info(f"Processing single region: {region_name}")

    logger.info(f"🎯 Target date: {target_date}")

    all_results: Dict[str, Any] = {}
    success_count = 0
    failure_count = 0

    for region in regions_to_process:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"📌 PROCESSING REGION: {region}")
        logger.info(f"{'=' * 80}")
        logger.debug(f"Using id chunk: {id_chunk_size}")
        logger.debug(f"With save interval {save_interval}")

        result = process_single_date_for_region(
            region=region,
            date_str=target_date,
            env_path=env_path,
            n_jobs=n_jobs,
            id_chunk_size=id_chunk_size,
            save_interval=save_interval
        )

        all_results[region] = result

        if result.get('success', False):
            success_count += 1
            logger.info(f"✅ Region {region} completed successfully")
            logger.info(f"   Breakpoints found: {result.get('breakpoints_found', 0):,}")
        else:
            failure_count += 1
            logger.warning(f"❌ Region {region} failed")
            if 'reason' in result:
                logger.warning(f"   Reason: {result['reason']}")

        time.sleep(3)

    logger.info(f"\n{'=' * 80}")
    logger.info("📊 FINAL SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Target date: {target_date}")
    logger.info(f"Total regions processed: {len(regions_to_process)}")
    logger.info(f"✅ Successful: {success_count}")
    logger.info(f"❌ Failed: {failure_count}")

    total_breakpoints_all = sum(
        r.get('breakpoints_found', 0)
        for r in all_results.values()
        if r.get('success', False)
    )
    logger.info(f"Total breakpoints found across all regions: {total_breakpoints_all:,}")

    logger.info(f"\n📋 Results by region:")
    for region, result in all_results.items():
        status = "✅" if result.get('success', False) else "❌"
        breakpoints = result.get('breakpoints_found', 0)
        total_ids = result.get('total_ids', 0)
        logger.info(f"  {status} {region}: {breakpoints:,} breakpoints from {total_ids:,} IDs")

    logger.info("=" * 80)
    logger.info("PROCESS_MISSING_NRT.py COMPLETED")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
