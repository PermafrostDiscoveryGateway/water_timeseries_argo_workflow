from near_real_time_grid_v2 import verify_downloads_complete, verify_process_complete, merge_near_real_time_region, \
    process_near_real_time_region_dates_zarr, download_near_real_time_region_dates, generate_expected_dates, \
    merge_near_real_time_region_v3_simple, process_region_direct, debug_historical_dates, find_matching_lake_ids, \
    compare_netcdf_files, verify_merged_netcdf, verify_merged_data, merge_new_results, is_all_new_data_in_file
import sys
from typing import List, Dict, Any, Optional
import shutil
import gc
import utils.download_new_dynamic_world_data as download_new_dynamic_world_data
from loguru import logger
from datetime import date, datetime, timedelta
from dotenv import load_dotenv
import os
import glob
import time
import pandas as pd
import utils.region_boundaries
from pathlib import Path
import xarray as xr
import numpy as np

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


def get_summer_months(year: int) -> List[str]:
    """
    Get summer months (June-September) for a given year.

    Args:
        year: The year to get summer months for

    Returns:
        List of date strings in "YYYY-MM" format
    """
    summer_months = [6, 7, 8, 9]
    return [f"{year}-{month:02d}" for month in summer_months]


def get_summer_dates_for_processing(year: int) -> List[pd.Timestamp]:
    """
    Get Timestamp objects for the first day of each summer month.

    Args:
        year: The year to get summer dates for

    Returns:
        List of pd.Timestamp objects
    """
    summer_months = [6, 7, 8, 9]
    return [pd.Timestamp(f"{year}-{month:02d}-01") for month in summer_months]


def check_data_availability_for_date(region: str, date_str: str, env_path: str = None) -> Dict[str, Any]:
    """
    Check if data is available for a specific region and date.

    Args:
        region: Region name
        date_str: Date in "YYYY-MM" format
        env_path: Optional path to .env file

    Returns:
        dict: Availability information
    """
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    dynamic_world_data_dir = os.environ.get('dynamic_world_data')
    if not dynamic_world_data_dir:
        return {'available': False, 'error': 'dynamic_world_data not set'}

    # Check if merged file exists for this region/date
    data_file = Path(dynamic_world_data_dir) / 'merge' / f'dw_{region}_{date_str}.nc'

    if not data_file.exists():
        return {
            'available': False,
            'file_exists': False,
            'date_str': date_str,
            'region': region,
            'message': f'No data file found for {region} {date_str}'
        }

    try:
        ds = xr.open_dataset(str(data_file))
        id_count = len(ds.id_geohash) if 'id_geohash' in ds.dims else 0
        date_count = len(ds.date) if 'date' in ds.dims else 0

        # Check if the specific date exists in the file
        has_date = False
        if 'date' in ds.coords and date_count > 0:
            dates_in_file = pd.to_datetime(ds.date.values)
            date_strings = [d.strftime("%Y-%m") for d in dates_in_file]
            has_date = date_str in date_strings

        ds.close()

        return {
            'available': id_count > 0 and has_date,
            'file_exists': True,
            'id_count': id_count,
            'date_count': date_count,
            'has_date': has_date,
            'date_str': date_str,
            'region': region,
            'file_path': str(data_file)
        }
    except Exception as e:
        return {
            'available': False,
            'file_exists': True,
            'error': str(e),
            'date_str': date_str,
            'region': region
        }


def process_summer_months_for_region(
        region: str,
        years: List[int],
        env_path: str = None,
        force_processing: bool = False,
        use_direct_method: bool = True
) -> Dict[str, Any]:
    """
    Process summer months (June-September) for a region over multiple years.

    Args:
        region: Region name
        years: List of years to process
        env_path: Optional path to .env file
        force_processing: If True, process even if data availability is low
        use_direct_method: If True, use process_region_direct (faster, more reliable).
                           If False, use process_near_real_time_region_dates_zarr (grid-based).

    Returns:
        dict: Processing results for each month
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"PROCESSING SUMMER MONTHS FOR REGION: {region}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Years to process: {years}")
    logger.info(f"Using direct processing method: {use_direct_method}")

    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    results = {
        'region': region,
        'years': years,
        'months_processed': [],
        'results': {},
        'use_direct_method': use_direct_method
    }

    for year in years:
        logger.info(f"\n--- Processing year: {year} ---")

        # Get summer months for this year
        summer_months = get_summer_months(year)
        summer_dates = get_summer_dates_for_processing(year)

        for month_str, timestamp in zip(summer_months, summer_dates):
            logger.info(f"\nChecking month: {month_str}")

            # Check if data is available
            availability = check_data_availability_for_date(region, month_str, env_path)

            if not availability.get('available', False):
                logger.warning(f"  ⚠️ No data available for {region} {month_str}")
                logger.warning(f"     Reason: {availability.get('message', availability.get('error', 'Unknown'))}")

                results['results'][month_str] = {
                    'success': False,
                    'reason': 'No data available',
                    'details': availability
                }
                continue

            logger.info(f"  ✅ Data available for {region} {month_str}")
            logger.info(f"     IDs in file: {availability.get('id_count', 0):,}")

            # Process the month
            try:
                logger.info(f"  Processing {region} for {month_str}...")

                if use_direct_method:
                    # Use the direct processing method (bypasses grid, uses IDs directly)
                    process_result = process_region_direct(
                        region=region,
                        analysis_date=month_str,  # Pass as string, not list
                        env_path=env_path,
                        batch_size=1000
                    )
                else:
                    # Use the original grid-based method
                    process_result = process_near_real_time_region_dates_zarr(
                        region=region,
                        current_analysis_dates=[timestamp],
                        env_path=env_path
                    )

                results['results'][month_str] = {
                    'success': process_result.get('success', False) if isinstance(process_result,
                                                                                  dict) else process_result,
                    'timestamp': timestamp,
                    'year': year,
                    'month': month_str,
                    'availability': availability
                }

                # Add additional info if direct method was used
                if use_direct_method and isinstance(process_result, dict):
                    results['results'][month_str].update({
                        'total_ids': process_result.get('total_ids', 0),
                        'processed': process_result.get('processed', 0),
                        'breakpoints_found': process_result.get('breakpoints_found', 0),
                        'zarr_path': process_result.get('zarr_path', None)
                    })

                if results['results'][month_str]['success']:
                    logger.info(f"  ✅ Successfully processed {region} {month_str}")
                    results['months_processed'].append(month_str)
                else:
                    logger.warning(f"  ❌ Failed to process {region} {month_str}")

            except Exception as e:
                logger.error(f"  ❌ Error processing {region} {month_str}: {e}")
                import traceback
                traceback.print_exc()
                results['results'][month_str] = {
                    'success': False,
                    'error': str(e),
                    'timestamp': timestamp,
                    'year': year,
                    'month': month_str,
                    'availability': availability
                }

            # Small delay between processing months to avoid memory issues
            time.sleep(2)

        logger.info(f"Completed year {year}")

    # Summary for the region
    processed_count = sum(1 for r in results['results'].values() if r.get('success', False))
    total_count = len(results['results'])

    logger.info(f"\n{'=' * 80}")
    logger.info(f"SUMMARY FOR REGION: {region}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total months processed: {total_count}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Failed/Skipped: {total_count - processed_count}")

    if processed_count > 0:
        success_rate = (processed_count / total_count) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")

        # Show breakpoints found if using direct method
        if use_direct_method:
            total_breakpoints = sum(
                r.get('breakpoints_found', 0)
                for r in results['results'].values()
                if r.get('success', False)
            )
            if total_breakpoints > 0:
                logger.info(f"Total breakpoints found: {total_breakpoints:,}")

    return results


def main():
    """Main function to process summer months for all regions."""

    logger.debug("=" * 80)
    logger.debug("PROCESS_REGION.PY STARTED (SUMMER MONTHS PROCESSING)")
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
    original_most_recent_dynamic_world_file = max(all_dynamic_world_files, key=lambda f: Path(f).stat().st_mtime)
    logger.debug(f"Dates in the historical file")
    debug_historical_dates(historical_file_path=original_most_recent_dynamic_world_file)

    for var in env_vars_to_check:
        value = os.environ.get(var, 'NOT SET')
        logger.debug(f"{var} = {value}")

    # Get region from environment or use all regions
    region_name = os.environ.get("region_name", "ALL")

    # If region is "ALL", get all regions from boundaries
    if region_name == "ALL":
        boundaries = utils.region_boundaries.get_region_boundaries()
        regions_to_process = list(boundaries.keys())
        logger.info(f"Processing ALL regions: {regions_to_process}")
    else:
        regions_to_process = [region_name]
        logger.info(f"Processing single region: {region_name}")

    # Determine which years to process
    current_year = datetime.now().year

    # Process current year and previous year
    years_to_process = [
        current_year - 2,  # Two years ago
        current_year - 1,  # Last year
        current_year  # Current year
    ]

    # Only include years that have passed (or current year)
    years_to_process = [y for y in years_to_process if y <= current_year]

    logger.info(f"Years to process: {years_to_process}")

    # Check if we should run based on date
    SHOULD_RUN = False
    summer_months = [6, 7, 8, 9]
    TODAY = datetime.now()
    TODAY_MONTH = TODAY.month

    # Always run if we're processing all regions (for testing/comparison)
    if region_name == "ALL":
        SHOULD_RUN = True
        logger.info("Running for all regions (forced)")
    elif TODAY_MONTH - 1 in summer_months:
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 3:
            SHOULD_RUN = True
            logger.debug(f"TODAY_DAY: {TODAY_DAY} - Should run: {SHOULD_RUN}")
    else:
        logger.debug(f"Current month {TODAY_MONTH} is not in summer months or too early")
        # Still run for historical data if we're processing all regions
        if region_name == "ALL":
            SHOULD_RUN = True
            logger.info("Running for all regions (historical data processing)")

    if not SHOULD_RUN:
        logger.info("Skipping processing - conditions not met")
        return

    # Process each region
    all_results = {}

    # Determine which regions to process with direct method vs grid method
    # Direct method is better for regions with sparse data or grid filtering issues
    direct_method_regions = ['EURASIA3', 'EURASIA2', 'TEST']  # Add regions that have issues with grid filtering
    use_direct_method = True  # Default to direct method

    for region in regions_to_process:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing region: {region}")
        logger.info(f"{'=' * 80}")

        # Use direct method for problematic regions, otherwise use grid method
        use_direct = region in direct_method_regions

        # Check if we have any data for this region
        test_date = f"{current_year}-06"
        availability = check_data_availability_for_date(region, test_date, env_path)

        if not availability.get('available', False) and region != "TEST":
            logger.warning(f"⚠️ Region {region} may not have data for summer months")
            logger.warning(f"   Availability check: {availability}")

            # Try previous year
            test_date_prev = f"{current_year - 1}-06"
            availability_prev = check_data_availability_for_date(region, test_date_prev, env_path)

            if not availability_prev.get('available', False):
                logger.warning(f"⚠️ Region {region} has no data for {test_date} or {test_date_prev}")
                logger.warning(f"   Skipping this region...")
                continue

        # Process summer months for this region
        try:
            result = process_summer_months_for_region(
                region=region,
                years=years_to_process,
                env_path=env_path,
                force_processing=False,
                use_direct_method=use_direct
            )
            all_results[region] = result

            # Log summary for this region
            processed_count = sum(1 for r in result['results'].values() if r.get('success', False))
            total_count = len(result['results'])
            method_used = "DIRECT" if use_direct else "GRID"
            logger.info(
                f"\n✅ Region {region} ({method_used}): {processed_count}/{total_count} months processed successfully")

            if use_direct:
                total_breakpoints = sum(
                    r.get('breakpoints_found', 0)
                    for r in result['results'].values()
                    if r.get('success', False)
                )
                if total_breakpoints > 0:
                    logger.info(f"   Total breakpoints found: {total_breakpoints:,}")

            # List which months succeeded
            successful_months = [m for m, r in result['results'].items() if r.get('success', False)]
            if successful_months:
                logger.info(f"   Successful months: {successful_months}")

        except Exception as e:
            logger.error(f"❌ Error processing region {region}: {e}")
            import traceback
            traceback.print_exc()
            all_results[region] = {'error': str(e)}

        # Small delay between regions
        time.sleep(3)

    # Final summary
    logger.info(f"\n{'=' * 80}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'=' * 80}")

    total_successful = 0
    total_attempted = 0
    total_breakpoints_all = 0

    for region, result in all_results.items():
        if 'results' in result:
            region_success = sum(1 for r in result['results'].values() if r.get('success', False))
            region_total = len(result['results'])
            total_successful += region_success
            total_attempted += region_total

            method_used = "DIRECT" if result.get('use_direct_method', False) else "GRID"

            # Count breakpoints if direct method
            if result.get('use_direct_method', False):
                region_breakpoints = sum(
                    r.get('breakpoints_found', 0)
                    for r in result['results'].values()
                    if r.get('success', False)
                )
                total_breakpoints_all += region_breakpoints
                logger.info(
                    f"  {region} ({method_used}): {region_success}/{region_total} months successful, {region_breakpoints:,} breakpoints")
            else:
                logger.info(f"  {region} ({method_used}): {region_success}/{region_total} months successful")
        else:
            logger.info(f"  {region}: Error - {result.get('error', 'Unknown error')}")

    if total_attempted > 0:
        success_rate = (total_successful / total_attempted) * 100
        logger.info(f"\nOverall success rate: {success_rate:.1f}% ({total_successful}/{total_attempted})")
        if total_breakpoints_all > 0:
            logger.info(f"Total breakpoints found across all regions: {total_breakpoints_all:,}")

    logger.info("=" * 80)
    logger.info("PROCESS_REGION.PY COMPLETED")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()