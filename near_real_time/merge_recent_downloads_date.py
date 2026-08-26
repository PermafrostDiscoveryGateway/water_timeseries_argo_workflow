"""Merge recent downloads for a single, explicitly-named month.

Unlike merge_recent_downloads.py (which always targets the previous month, gated to
only run during the summer processing window), this script takes an explicit target
month (e.g. "2026-06") as a command-line argument, so it can be used to backfill/fix
a month that was missed or came out incomplete.

Usage:
    python merge_recent_downloads_date.py 2026-06 [path/to/.env]
"""
from merge_recent_downloads import (
    _configure_dask_for_low_memory,
    process_region_fast,
    combine_region_files,
    verify_combined_file_optimized,
)
import sys
import re
import os
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

DATE_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def main():
    logger.debug("=" * 80)
    logger.debug("MERGE_RECENT_DOWNLOADS_DATE.PY STARTED")
    logger.debug("=" * 80)

    if len(sys.argv) < 2:
        logger.error('Usage: python merge_recent_downloads_date.py <YYYY-MM> [path/to/.env]')
        sys.exit(1)

    date_to_run = sys.argv[1]
    if not DATE_PATTERN.match(date_to_run):
        logger.error(f"Invalid date '{date_to_run}' - expected format YYYY-MM, e.g. 2026-06")
        sys.exit(1)

    _configure_dask_for_low_memory()

    env_path = None
    if len(sys.argv) > 2:
        env_path = sys.argv[2]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    import utils.region_boundaries
    boundaries = utils.region_boundaries.get_region_boundaries()
    all_regions = list(boundaries.keys())
    logger.info(f"Available regions: {all_regions}")

    dynamic_world_data_dir = os.environ['dynamic_world_data']

    logger.info(f"🎯 Target date: {date_to_run}")

    results = {}
    success_count = 0
    failure_count = 0
    skipped_count = 0
    partial_count = 0
    region_files = []

    for region in all_regions:
        try:
            logger.info(f"\n{'#' * 80}")
            logger.info(f"PROCESSING REGION: {region}")
            logger.info(f"{'#' * 80}")

            result = process_region_fast(
                region=region,
                date_to_run=date_to_run,
                env_path=env_path,
                dynamic_world_data_dir=dynamic_world_data_dir
            )

            results[region] = result

            if result.get('success', False):
                if result.get('partial', False):
                    partial_count += 1
                    logger.info(f"⚠️ Region {region} processed partially (acceptable)")
                elif result.get('reason') == 'Already merged':
                    skipped_count += 1
                    logger.info(f"⏭️ Region {region} already merged (skipped)")
                else:
                    success_count += 1
                    logger.info(f"✅ Region {region} processed successfully!")

                if 'merged_file' in result:
                    region_files.append(result['merged_file'])
            else:
                failure_count += 1
                logger.error(f"❌ Region {region} failed: {result.get('reason', 'Unknown error')}")

        except Exception as e:
            logger.error(f"❌ Error processing region {region}: {e}")
            import traceback
            traceback.print_exc()
            results[region] = {
                'region': region,
                'date': date_to_run,
                'success': False,
                'reason': f'Exception: {str(e)}',
                'error': str(e)
            }
            failure_count += 1

    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY - ALL REGIONS (MERGE FOR EXPLICIT DATE)")
    logger.info("=" * 80)
    logger.info(f"Date processed: {date_to_run}")
    logger.info(f"Total regions: {len(all_regions)}")
    logger.info(f"✅ Fully successful: {success_count}")
    logger.info(f"⚠️ Partial (acceptable): {partial_count}")
    logger.info(f"⏭️ Already merged: {skipped_count}")
    logger.info(f"❌ Failed: {failure_count}")
    logger.info("=" * 80)

    if failure_count == 0 and region_files:
        logger.info("\n" + "=" * 80)
        logger.info("COMBINING ALL REGION FILES")
        logger.info("=" * 80)
        logger.info(f"Found {len(region_files)} region files to combine")

        expected_files = [f"dw_{region}_{date_to_run}.nc" for region in all_regions]
        missing_files = [f for f in expected_files if f not in [Path(f).name for f in region_files]]

        if missing_files:
            logger.warning(f"⚠️ Missing region files: {missing_files}")
            logger.warning("Skipping combination due to missing files")
        else:
            combined_file_name = f"dynamic_world_combined_{date_to_run}.nc"
            combined_file_path = os.path.join(dynamic_world_data_dir, 'merge', combined_file_name)

            logger.info(f"Combining into: {combined_file_path}")

            combine_result = combine_region_files(
                region_files=region_files,
                output_file=combined_file_path,
                env_path=env_path
            )

            if combine_result.get('success', False):
                logger.info(f"✅ Combined file created successfully!")

                logger.info("\n" + "=" * 80)
                logger.info("VERIFYING COMBINED FILE")
                logger.info("=" * 80)

                verify_result = verify_combined_file_optimized(
                    combined_file_path=combined_file_path,
                    regions=all_regions,
                    date_to_check=date_to_run,
                    env_path=env_path,
                    sample_size=1000
                )

                if verify_result.get('success', False):
                    if verify_result.get('all_regions_complete', True):
                        logger.info("\n" + "=" * 80)
                        logger.info("✅ ALL REGIONS VERIFIED SUCCESSFULLY!")
                        logger.info("=" * 80)
                        logger.info(f"  Combined file: {combined_file_path}")
                        logger.info(f"  Total IDs: {verify_result['total_ids']:,}")
                        logger.info(f"  Date: {date_to_run}")
                        logger.info("=" * 80)
                    else:
                        logger.warning("\n" + "=" * 80)
                        logger.warning("⚠️ SOME REGIONS HAVE PARTIAL DATA (but acceptable)")
                        logger.warning("=" * 80)
                        logger.info(f"  Combined file: {combined_file_path}")
                        logger.info(f"  Total IDs: {verify_result['total_ids']:,}")
                        logger.info(f"  Date: {date_to_run}")

                        for region, reg_result in verify_result.get('region_results', {}).items():
                            if reg_result.get('partial', False):
                                completion_pct = reg_result.get('completion_pct', 0)
                                logger.warning(f"  Region {region}: {completion_pct:.2%} complete (acceptable)")

                        logger.info("=" * 80)
                else:
                    logger.warning("\n" + "=" * 80)
                    logger.warning("⚠️ VERIFICATION ISSUES FOUND")
                    logger.warning("=" * 80)

                    for region, reg_result in verify_result.get('region_results', {}).items():
                        if not reg_result.get('success', False):
                            logger.warning(f"  Region {region}: {reg_result.get('error', 'Unknown error')}")
                            if 'ids_missing' in reg_result:
                                logger.warning(
                                    f"    {reg_result['ids_missing']} IDs missing ({reg_result.get('completion_pct', 0):.2%} complete)")
                                if 'missing_sample' in reg_result:
                                    logger.warning(f"    Sample missing IDs: {reg_result['missing_sample']}")

                    logger.warning("=" * 80)
            else:
                logger.error(f"❌ Failed to combine region files: {combine_result.get('error', 'Unknown error')}")
    else:
        if failure_count > 0:
            logger.warning("⚠️ Not combining region files due to failures")
        elif not region_files:
            logger.warning("⚠️ No region files to combine")

    logger.info("\n" + "=" * 80)
    logger.info("SCRIPT COMPLETED")
    logger.info("=" * 80)

    sys.exit(0 if failure_count == 0 else 1)


if __name__ == "__main__":
    main()
