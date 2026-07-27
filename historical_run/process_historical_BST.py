from utils.helper_functions import debug_historical_dates, process_region_date_new_fast_historical_safe
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
    """
    Check if data exists for a specific region and date.

    Args:
        region: Region name (e.g., "TEST", "AFRICA")
        date_str: Date in "YYYY-MM" format
        env_path: Optional path to .env file

    Returns:
        dict: Availability status and metadata
    """
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    dynamic_world_data_dir = os.environ.get('dynamic_world_data')
    if not dynamic_world_data_dir:
        return {'available': False, 'error': 'dynamic_world_data not set'}

    # 1. Check if merged file exists in merge directory
    data_file = Path(dynamic_world_data_dir) / 'merge' / f'dw_{region}_{date_str}.nc'
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
            elif 'id' in ds.dims:
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
                has_date = date_str in date_strings
            elif 'time' in ds.coords:
                dates_in_file = pd.to_datetime(ds.time.values)
                date_strings = [d.strftime("%Y-%m") for d in dates_in_file]
                has_date = date_str in date_strings
            else:
                # If there's only one date, assume it's the right one
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
                logger.info(f"File exists but failed checks: has_ids={has_ids}, has_date={has_date}")

        except Exception as e:
            logger.debug(f"Error checking merge file: {e}")
            logger.debug(f"Full error: {traceback.format_exc()}")

    # 2. Check historical file
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
        n_jobs: int = 1,  # Keep at 1 for RBEAST stability
        id_chunk_size: int = 50,  # Reduced for RBEAST
        save_interval: int = 1,
        beast_kwargs: Dict[str, Any] = None,
        break_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Process a single date for a region using the SAFE historical method.

    This version uses individual lake processing to avoid segmentation faults
    in RBEAST (C++ extension).

    Args:
        region: Region name
        date_str: Date in "YYYY-MM" format
        env_path: Optional path to .env file
        n_jobs: Number of parallel jobs (must be 1 for RBEAST)
        id_chunk_size: Number of IDs to process per chunk (recommended: 1-50)
        save_interval: Save intermediate results every N chunks
        beast_kwargs: Optional RBEAST parameters
        break_threshold: Probability threshold for break detection (default: 0.5)

    Returns:
        dict: Processing results
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"PROCESSING {region} FOR {date_str} (SAFE HISTORICAL METHOD)")
    logger.info(f"{'=' * 80}")

    # Load environment if env_path provided
    if env_path:
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    # Validate n_jobs - must be 1 for RBEAST stability
    if n_jobs > 1:
        logger.warning(f"n_jobs={n_jobs} > 1 is not recommended for RBEAST. Forcing n_jobs=1 for safety.")
        n_jobs = 1

    # Validate id_chunk_size - small chunks for RBEAST
    if id_chunk_size > 100:
        logger.warning(f"id_chunk_size={id_chunk_size} is large for RBEAST. Consider reducing to 1-50 for stability.")
        # Don't force change, but warn user

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

    # Default RBEAST parameters for stability
    if beast_kwargs is None:
        beast_kwargs = {
            'trendMaxOrder': 0,
            'trendMinSepDist': 1,
            'nCpMax': 3,  # Limit maximum number of change points
            'maxK': 2,  # Reduce model complexity
        }

    # Process using SAFE method (individual lakes)
    try:
        logger.info(f"🚀 Processing {region} for {date_str} with SAFE method...")
        logger.info(f"   Each chunk: {id_chunk_size} IDs (processed individually for RBEAST)")
        logger.info(f"   RBEAST parameters: {beast_kwargs}")
        logger.info(f"   Break threshold: {break_threshold}")
        logger.info(f"   Save interval: every {save_interval} chunks")

        process_result = process_region_date_new_fast_historical_safe(
            region=region,
            analysis_date=date_str,
            env_path=env_path,
            n_jobs=n_jobs,
            id_chunk_size=id_chunk_size,
            save_interval=save_interval,
            beast_kwargs=beast_kwargs,
            break_threshold=break_threshold,
            debug_mode=True
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
    """Main function to process historical dates with RBEAST."""

    logger.debug("=" * 80)
    logger.debug("PROCESS_HISTORICAL_RBEAST.PY STARTED")
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

    dynamic_world_data_dir = os.environ.get('dynamic_world_data')
    if dynamic_world_data_dir:
        original_most_recent_dynamic_world_file = os.path.join(
            dynamic_world_data_dir, 'lakes_dw_V2d_compressed.nc'
        )
        logger.debug(f"Dates in the historical file")
        debug_historical_dates(historical_file_path=original_most_recent_dynamic_world_file)

    for var in env_vars_to_check:
        value = os.environ.get(var, 'NOT SET')
        logger.debug(f"{var} = {value}")

    # Get region from environment or use specific regions
    region_name = os.environ.get("region_name", "ALL")

    # SAFE defaults for RBEAST
    id_chunk_size = int(os.environ.get("id_chunk_size", 50))  # Small chunks
    save_interval = int(os.environ.get("save_interval", 1))
    n_jobs = 1  # Force to 1 for RBEAST

    # RBEAST parameters
    break_threshold = float(os.environ.get("break_threshold", 0.5))

    # Define regions to process
    if region_name == "ALL":
        # All main regions - can be customized
        regions_to_process = ['TEST', 'ALASKA', 'CANADA', 'EURASIA1', 'EURASIA2', 'EURASIA3']
        logger.info(f"Processing ALL main regions: {regions_to_process}")
    else:
        regions_to_process = [region_name]
        logger.info(f"Processing single region: {region_name}")

    # Target date (can be modified)
    target_date = "2025-06"
    logger.info(f"🎯 Target date: {target_date}")

    # Check if we should run based on date
    SHOULD_RUN = False
    summer_months = [6, 7, 8, 9]
    TODAY = datetime.now()
    TODAY_MONTH = TODAY.month

    # Always run if we're processing all regions or if it's summer
    if region_name == "ALL":
        SHOULD_RUN = True
        logger.info("Running for all regions (forced)")
    elif TODAY_MONTH - 1 in summer_months:
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 3:
            SHOULD_RUN = True
            logger.debug(f"TODAY_DAY: {TODAY_DAY} - Should run: {SHOULD_RUN}")
    else:
        # Still run for testing historical data
        if region_name != "ALL":
            SHOULD_RUN = True
            logger.info("Running for single region (testing mode)")

    if not SHOULD_RUN:
        logger.info("Skipping processing - conditions not met")
        return

    # Process each region
    all_results = {}
    success_count = 0
    failure_count = 0

    for region in regions_to_process:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"📌 PROCESSING REGION: {region}")
        logger.info(f"{'=' * 80}")
        logger.debug(f"Using id chunk: {id_chunk_size}")
        logger.debug(f"With save interval {save_interval}")

        # Process the single date with safe method
        result = process_single_date_for_region(
            region=region,
            date_str=target_date,
            env_path=env_path,
            n_jobs=n_jobs,
            id_chunk_size=id_chunk_size,
            save_interval=save_interval,
            beast_kwargs={
                'trendMaxOrder': 0,
                'trendMinSepDist': 1,
                'nCpMax': 3,
                'maxK': 2,
            },
            break_threshold=break_threshold
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
    logger.info("PROCESS_HISTORICAL_RBEAST.PY COMPLETED")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()