from utils.helper_functions import process_region_date_new_fast_NRT, debug_historical_dates
from utils.date_gate import is_test_run, most_recent_summer_month
import sys
from typing import List, Dict, Any, Optional
import gc
import traceback
from loguru import logger
from datetime import datetime
from dotenv import load_dotenv
import os
import glob
import time
import pandas as pd
from pathlib import Path
import xarray as xr
# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def is_file_ready(filepath, wait_seconds=0.5, checks=20):
    """Check if a file is ready (not being written to)."""
    sizes = []
    for _ in range(checks):
        size = os.path.getsize(filepath)
        sizes.append(size)
        time.sleep(wait_seconds)
    # If size hasn't changed, assume writing is done
    return len(set(sizes)) == 1


def check_data_availability_for_date(region: str, date_str: str, env_path: str = None) -> Dict[str, Any]:
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    dynamic_world_data_dir = os.environ.get('dynamic_world_data')
    if not dynamic_world_data_dir:
        return {'available': False, 'error': 'dynamic_world_data not set'}

    # 1. Check if merged file exists in merge directory
    # Try both the original date_str and the zero-padded version
    date_variations = [date_str]

    # If the date string has a single-digit month, also try zero-padded version
    if '-' in date_str:
        year, month = date_str.split('-')
        if len(month) == 1:
            # Add zero-padded version
            date_variations.append(f"{year}-{month.zfill(2)}")
        elif len(month) == 2 and month.startswith('0'):
            # If it's zero-padded, also try without leading zero
            date_variations.append(f"{year}-{month.lstrip('0')}")

    # Check each date variation
    for date_variant in date_variations:
        data_file = Path(dynamic_world_data_dir) / 'merge' / f'dw_{region}_{date_variant}.nc'
        logger.debug(f"Checking if file {data_file} exists: {os.path.exists(data_file)}")

        if data_file.exists():
            try:
                ds = xr.open_dataset(str(data_file))

                # Check for id_geohash in dims OR coordinates
                has_ids = False
                id_count = 0

                if 'id_geohash' in ds.dims:
                    id_count = len(ds.id_geohash)
                    has_ids = id_count > 0
                elif 'id_geohash' in ds.coords:
                    id_count = len(ds.id_geohash)
                    has_ids = id_count > 0
                elif 'id' in ds.dims:  # Alternative dimension name
                    id_count = len(ds.id)
                    has_ids = id_count > 0
                else:
                    # Try to find any dimension that might be IDs
                    for dim in ds.dims:
                        if 'id' in dim.lower():
                            id_count = len(ds[dim])
                            has_ids = True
                            break

                # Check if this file has the date we want
                has_date = False
                if 'date' in ds.coords:
                    dates_in_file = pd.to_datetime(ds.date.values)
                    date_strings = [d.strftime("%Y-%m") for d in dates_in_file]
                    has_date = date_str in date_strings or date_variant in date_strings
                elif 'time' in ds.coords:  # Alternative coordinate name
                    dates_in_file = pd.to_datetime(ds.time.values)
                    date_strings = [d.strftime("%Y-%m") for d in dates_in_file]
                    has_date = date_str in date_strings or date_variant in date_strings
                else:
                    # If there's only one date, assume it's the right one
                    # This is likely a single-date file
                    has_date = True

                ds.close()

                if has_ids and has_date:
                    return {
                        'available': True,
                        'file_exists': True,
                        'id_count': id_count,
                        'date_count': 1,
                        'has_date': has_date,
                        'date_str': date_str,
                        'region': region,
                        'file_path': str(data_file),
                        'source': 'merge_directory'
                    }
                else:
                    # Debug info
                    logger.info(f"File exists but failed checks: has_ids={has_ids}, has_date={has_date}")

            except Exception as e:
                logger.debug(f"Error checking merge file: {e}")
                logger.debug(f"Full error: {traceback.format_exc()}")

    # 2. Check historical file (this part remains unchanged)
    historical_file = Path(dynamic_world_data_dir) / 'lakes_dw_V2d_compressed.nc'
    if historical_file.exists():
        try:
            ds = xr.open_dataset(str(historical_file))

            # Check if date exists
            has_date = False
            id_count = 0

            if 'date' in ds.coords:
                dates_in_file = pd.to_datetime(ds.date.values)
                date_strings = [d.strftime("%Y-%m") for d in dates_in_file]
                has_date = date_str in date_strings

                # Get ID count for the region
                if 'id_geohash' in ds.dims:
                    id_count = len(ds.id_geohash)
                elif 'id' in ds.dims:
                    id_count = len(ds.id)

            ds.close()

            if has_date:
                return {
                    'available': True,
                    'file_exists': True,
                    'id_count': id_count,
                    'date_count': 1,
                    'has_date': True,
                    'date_str': date_str,
                    'region': region,
                    'file_path': str(historical_file),
                    'source': 'historical_file'
                }
        except Exception as e:
            logger.debug(f"Error checking historical file: {e}")

    # If we get here, no data was found
    return {
        'available': False,
        'file_exists': False,
        'date_str': date_str,
        'region': region,
        'message': f'No data found for {region} {date_str} in either source'
    }
def process_single_date_for_region(
        region: str,
        date_str: str,
        env_path: str = None,
        n_jobs: int = 12,
        id_chunk_size: int = 2000,  # Number of IDs per chunk (passed to calculate_break)
        save_interval: int = 10,   # Save every N chunks
) -> Dict[str, Any]:
    """
    Process a single date for a region using the FAST method.

    Args:
        region: Region name
        date_str: Date in "YYYY-MM" format
        env_path: Optional path to .env file
        n_jobs: Number of parallel jobs (passed to NRTBreakpoint internally)
        id_chunk_size: Number of IDs to process per chunk (default: 100)
        save_interval: Save intermediate results every N chunks (default: 10)

    Returns:
        dict: Processing results
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"PROCESSING {region} FOR {date_str} (FAST METHOD)")
    logger.info(f"{'=' * 80}")

    # Load environment if env_path provided
    if env_path:
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    # Check if data is available
    availability = check_data_availability_for_date(region, date_str, env_path)

    if not availability.get('available', False):
        logger.warning(f"⚠️ No data available for {region} {date_str}")
        logger.warning(f"   Reason: {availability.get('message', availability.get('error', 'Unknown'))}")
        return {
            'success': False,
            'region': region,
            'date': date_str,
            'reason': 'No data available',
            'details': availability
        }

    # Log the source of the data
    source = availability.get('source', 'unknown')
    logger.info(f"✅ Data available for {region} {date_str} (source: {source})")
    logger.info(f"   IDs in file: {availability.get('id_count', 0):,}")

    # Process using FAST method
    try:
        logger.info(f"🚀 Processing {region} for {date_str} with {n_jobs} parallel jobs...")
        logger.info(f"   Each chunk: {id_chunk_size} IDs")
        logger.info(f"   Save interval: every {save_interval} chunks ({save_interval * id_chunk_size} IDs)")

        process_result = process_region_date_new_fast_NRT(
            region=region,
            analysis_date=date_str,
            env_path=env_path,
            n_jobs=n_jobs,
            id_chunk_size=id_chunk_size,
            save_interval=save_interval
        )

        if process_result.get('success', False):
            logger.info(f"✅ Successfully processed {region} {date_str}")
            logger.info(f"   Total IDs: {process_result.get('total_ids', 0):,}")
            logger.info(f"   Breakpoints found: {process_result.get('breakpoints_found', 0):,}")
            logger.info(f"   Zarr path: {process_result.get('zarr_path', 'N/A')}")
        else:
            logger.warning(f"❌ Failed to process {region} {date_str}")

        return process_result

    except Exception as e:
        logger.error(f"❌ Error processing {region} {date_str}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'region': region,
            'date': date_str,
            'error': str(e)
        }


def main():
    """Main function to process June 2026 for all regions."""

    logger.debug("=" * 80)
    logger.debug("PROCESS_REGION.PY STARTED (JUNE 2026 ONLY)")
    logger.debug("=" * 80)

    # Load environment
    env_path = None
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    # Log all important environment variables
    env_vars_to_check = [
        'dynamic_world_data',
        'dynamic_world_downloads',
        'vector_lake_file',
        'output_dir',
        'project',
        'region_name'
    ]

    dynamic_world_data_dir = os.environ['dynamic_world_data']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
    if not all_dynamic_world_files:
        logger.error(f"No .nc files found in {dynamic_world_data_dir}")
        return {'success': False, 'error': 'No .nc files found'}

    original_most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getmtime)
    logger.debug(f"Most recent dynamic world file: {original_most_recent_dynamic_world_file}")
    time.sleep(10)
    logger.debug(f"Dates in the historical file")
    debug_historical_dates(historical_file_path=original_most_recent_dynamic_world_file)

    for var in env_vars_to_check:
        value = os.environ.get(var, 'NOT SET')
        logger.debug(f"{var} = {value}")

    # Get region from environment or use specific regions
    region_name = os.environ.get("region_name", "ALL")

    id_chunk_size = int(os.environ.get("id_chunk_size", 500))
    save_interval = int(os.environ.get("save_interval", 1))
    n_jobs = int(os.environ.get("n_jobs", 1))

    # Define regions to process - you can customize this list
    if region_name == "ALL":
        # All regions except TEST_v1, CANADA_v1, etc. (only the main ones)
        regions_to_process = ['TEST', 'ALASKA', 'CANADA', 'EURASIA1', 'EURASIA2', 'EURASIA3']
        logger.info(f"Processing ALL main regions: {regions_to_process}")
    else:
        regions_to_process = [region_name]
        logger.info(f"Processing single region: {region_name}")

    # ONLY process June 2026
    target_date = "2026-06"
    logger.info(f"🎯 Default target date: {target_date}")

    # Check if we should run based on date
    SHOULD_RUN = False
    summer_months = [6, 7, 8, 9]
    TODAY = datetime.now()
    TODAY_MONTH = TODAY.month
    TODAY_YEAR = TODAY.year

    if is_test_run():
        SHOULD_RUN = True
        target_date = most_recent_summer_month(TODAY).strftime("%Y-%m")
        logger.debug(f"test_run=True - bypassing day-of-month/season gate, using {target_date}")
    elif (TODAY_MONTH -1) in summer_months:
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 3:
            SHOULD_RUN = True
            target_month = TODAY_MONTH - 1
            target_date = f"{TODAY_YEAR}-{target_month:02d}"
            logger.debug(f"TODAY_DAY: {TODAY_DAY} - Should run: {SHOULD_RUN}")
            logger.debug(f"Target date: {target_date}")

    if not SHOULD_RUN:
        logger.info("Skipping processing - conditions not met")
        return

    # Always run if we're processing all regions or if it's summer


    # Process each region for June 2026
    all_results = {}
    success_count = 0
    failure_count = 0

    for region in regions_to_process:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"📌 PROCESSING REGION: {region}")
        logger.info(f"{'=' * 80}")
        logger.debug(f"Using id chunk: {id_chunk_size}")
        logger.debug(f"With save interval {save_interval}")

        # Process the single date
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

        # Small delay between regions
        time.sleep(3)

    # Final summary
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

    # List results by region
    logger.info(f"\n📋 Results by region:")
    for region, result in all_results.items():
        status = "✅" if result.get('success', False) else "❌"
        breakpoints = result.get('breakpoints_found', 0)
        total_ids = result.get('total_ids', 0)
        logger.info(f"  {status} {region}: {breakpoints:,} breakpoints from {total_ids:,} IDs")

    logger.info("=" * 80)
    logger.info("PROCESS_NRT.py COMPLETED")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()