from utils.helper_functions import debug_historical_dates, process_region_date_beast_historical
import sys
from typing import List, Dict, Any, Optional
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




def generate_date_range_from_env(
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
        months: List[int] = [6, 7, 8, 9]  # Summer months by default
) -> List[str]:
    """
    Generate a list of dates in "YYYY-MM" format for specified months
    between start and end dates (inclusive).

    Args:
        start_year: Starting year (e.g., 2015)
        start_month: Starting month (e.g., 6 for June)
        end_year: Ending year (e.g., 2026)
        end_month: Ending month (e.g., 9 for September)
        months: List of months to include (default: [6, 7, 8, 9])

    Returns:
        List of date strings in "YYYY-MM" format

    Example:
        >>> generate_date_range_from_env(2015, 6, 2026, 9)
        ['2015-06', '2015-07', '2015-08', '2015-09', '2016-06', ...]
    """
    dates = []

    # Validate inputs
    if start_year > end_year:
        raise ValueError(f"start_year ({start_year}) cannot be greater than end_year ({end_year})")

    if start_year == end_year and start_month > end_month:
        raise ValueError(
            f"start_month ({start_month}) cannot be greater than end_month ({end_month}) when years are equal")

    for year in range(start_year, end_year + 1):
        for month in months:
            # Skip if month is before start_month in the start year
            if year == start_year and month < start_month:
                continue

            # Skip if month is after end_month in the end year
            if year == end_year and month > end_month:
                continue

            # Format as "YYYY-MM" with zero-padded month
            dates.append(f"{year}-{month:02d}")

    return dates


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

    # Check historical file
    dynamic_world_data_dir = os.environ['dynamic_world_data']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
    if not all_dynamic_world_files:
        logger.error(f"No .nc files found in {dynamic_world_data_dir}")
        return {'success': False, 'error': 'No .nc files found'}

    historical_file = max(all_dynamic_world_files, key=os.path.getctime)
    if Path(historical_file).exists():
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
        n_jobs: int = 8,
        id_chunk_size: int = 500,
        save_interval: int = 1,
        beast_kwargs: Dict[str, Any] = None,
        break_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Process a single date for a region using the BEAST historical method.

    This version uses the new process_region_date_beast_historical function
    which processes lakes in chunks with parallelization.

    Args:
        region: Region name
        date_str: Date in "YYYY-MM" format
        env_path: Optional path to .env file
        n_jobs: Number of parallel jobs (default: 8)
        id_chunk_size: Number of IDs to process per chunk (default: 500)
        save_interval: Save intermediate results every N chunks
        beast_kwargs: Optional RBEAST parameters
        break_threshold: Probability threshold for break detection (default: 0.5)

    Returns:
        dict: Processing results
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"PROCESSING {region} FOR {date_str} (BEAST HISTORICAL METHOD)")
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

    # Default RBEAST parameters for stability
    if beast_kwargs is None:
        beast_kwargs = {
            'trendMaxOrder': 0,
            'trendMinSepDist': 1,
            'nCpMax': 3,
            'maxK': 2,
        }

    # Process using BEAST method
    try:
        logger.info(f"🚀 Processing {region} for {date_str} with BEAST method...")
        logger.info(f"   Each chunk: {id_chunk_size} IDs")
        logger.info(f"   Parallel jobs: {n_jobs}")
        logger.info(f"   RBEAST parameters: {beast_kwargs}")
        logger.info(f"   Break threshold: {break_threshold}")
        logger.info(f"   Save interval: every {save_interval} chunks")

        process_result = process_region_date_beast_historical(
            region=region,
            analysis_date=date_str,
            env_path=env_path,
            n_jobs=n_jobs,
            id_chunk_size=id_chunk_size,
            save_interval=save_interval,
            beast_kwargs=beast_kwargs,
            break_threshold=break_threshold
        )

        if process_result.get('success', False):
            logger.info(f"✅ Successfully processed {region} {date_str}")
            logger.info(f"   Total IDs: {process_result.get('total_ids', 0):,}")
            logger.info(f"   Breakpoints found: {process_result.get('breakpoints_found', 0):,}")
            logger.info(f"   Zarr path: {process_result.get('zarr_path', 'N/A')}")
        else:
            logger.warning(f"❌ Failed to process {region} {date_str}")
            if 'error' in process_result:
                logger.warning(f"   Error: {process_result['error']}")

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
    """Main function to process historical dates with RBEAST using the new method."""

    logger.debug("=" * 80)
    logger.debug("PROCESS_HISTORICAL_RBEAST.PY STARTED (NEW BEAST METHOD)")
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

    original_most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
    if dynamic_world_data_dir and original_most_recent_dynamic_world_file:
        logger.debug(f"Dates in the historical file")
        debug_historical_dates(historical_file_path=original_most_recent_dynamic_world_file)

    for var in env_vars_to_check:
        value = os.environ.get(var, 'NOT SET')
        logger.debug(f"{var} = {value}")

    # Get date range from environment
    start_year = int(os.environ.get("start_year", 2015))
    start_month = int(os.environ.get("start_month", 6))
    end_year = int(os.environ.get("end_year", 2026))
    end_month = int(os.environ.get("end_month", 9))

    # Optional: Allow custom months via environment
    months_str = os.environ.get("months_to_process", "6,7,8,9")
    months = [int(m.strip()) for m in months_str.split(",")]

    # Generate dates
    dates_to_process = generate_date_range_from_env(
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        months=months
    )

    logger.info(f"Generated {len(dates_to_process)} dates to process")
    logger.info(f"Date range: {dates_to_process[0]} to {dates_to_process[-1]}")
    logger.info(f"Months included: {months}")

    # Get region from environment or use specific regions
    region_name = os.environ.get("region_name", "TEST")

    # Configuration for BEAST processing
    id_chunk_size = int(os.environ.get("id_chunk_size", 500))
    save_interval = int(os.environ.get("save_interval", 1))
    n_jobs = int(os.environ.get("n_jobs", 8))

    # RBEAST parameters
    break_threshold = float(os.environ.get("break_threshold", 0.5))

    # Optional: Allow customization of beast_kwargs via environment
    beast_kwargs = {
        'trendMaxOrder': int(os.environ.get("beast_trendMaxOrder", 0)),
        'trendMinSepDist': int(os.environ.get("beast_trendMinSepDist", 1)),
        'nCpMax': int(os.environ.get("beast_nCpMax", 3)),
        'maxK': int(os.environ.get("beast_maxK", 2)),
    }

    # Define regions to process
    if region_name == "ALL":
        # All main regions - can be customized
        regions_to_process = ['TEST', 'ALASKA', 'CANADA', 'EURASIA1', 'EURASIA2', 'EURASIA3']
        logger.info(f"Processing ALL main regions: {regions_to_process}")
    else:
        regions_to_process = [region_name]
        logger.info(f"Processing single region: {region_name}")

    # Process each region for each date
    all_results = {}
    success_count = 0
    failure_count = 0
    total_breakpoints_all = 0
    total_ids_all = 0

    for region in regions_to_process:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"📌 PROCESSING REGION: {region}")
        logger.info(f"{'=' * 80}")

        # Process all dates for this region
        region_results = {}

        for date_idx, target_date in enumerate(dates_to_process):
            logger.info(f"\n{'─' * 60}")
            logger.info(f"📅 Processing date {date_idx + 1}/{len(dates_to_process)}: {target_date}")
            logger.info(f"{'─' * 60}")
            logger.debug(f"Using id chunk: {id_chunk_size}")
            logger.debug(f"With save interval {save_interval}")
            logger.debug(f"Using {n_jobs} parallel jobs")

            # Process the date with BEAST method
            result = process_single_date_for_region(
                region=region,
                date_str=target_date,
                env_path=env_path,
                n_jobs=n_jobs,
                id_chunk_size=id_chunk_size,
                save_interval=save_interval,
                beast_kwargs=beast_kwargs,
                break_threshold=break_threshold
            )

            key = f"{region}_{target_date}"
            region_results[target_date] = result
            all_results[key] = result

            if result.get('success', False):
                success_count += 1
                breakpoints = result.get('breakpoints_found', 0)
                total_ids = result.get('total_ids', 0)
                total_breakpoints_all += breakpoints
                total_ids_all += total_ids
                logger.info(f"✅ {region} {target_date} completed successfully")
                logger.info(f"   Breakpoints found: {breakpoints:,}")
                logger.info(f"   Total IDs: {total_ids:,}")
            else:
                failure_count += 1
                logger.warning(f"❌ {region} {target_date} failed")
                if 'reason' in result:
                    logger.warning(f"   Reason: {result['reason']}")
                if 'error' in result:
                    logger.warning(f"   Error: {result['error']}")

            # Small delay between dates to avoid overwhelming resources
            if date_idx < len(dates_to_process) - 1:
                time.sleep(2)

        # Small delay between regions
        if regions_to_process.index(region) < len(regions_to_process) - 1:
            time.sleep(3)

    # Final summary
    logger.info(f"\n{'=' * 80}")
    logger.info("📊 FINAL SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Date range: {dates_to_process[0]} to {dates_to_process[-1]}")
    logger.info(f"Total dates processed: {len(dates_to_process)}")
    logger.info(f"Total regions processed: {len(regions_to_process)}")
    logger.info(f"Total region-date combinations: {len(all_results)}")
    logger.info(f"✅ Successful: {success_count}")
    logger.info(f"❌ Failed: {failure_count}")
    logger.info(f"Total breakpoints found across all regions and dates: {total_breakpoints_all:,}")
    logger.info(f"Total IDs processed across all regions and dates: {total_ids_all:,}")

    # List results by region and date
    logger.info(f"\n📋 Detailed results by region and date:")
    for region in regions_to_process:
        logger.info(f"\n  Region: {region}")
        region_success = 0
        region_fail = 0
        for target_date in dates_to_process:
            key = f"{region}_{target_date}"
            result = all_results.get(key, {})
            status = "✅" if result.get('success', False) else "❌"
            breakpoints = result.get('breakpoints_found', 0)
            total_ids = result.get('total_ids', 0)
            zarr_path = result.get('zarr_path', 'N/A')

            if result.get('success', False):
                region_success += 1
                logger.info(f"    {status} {target_date}: {breakpoints:,} breakpoints from {total_ids:,} IDs")
                logger.info(f"       Zarr: {zarr_path}")
            else:
                region_fail += 1
                reason = result.get('reason', result.get('error', 'Unknown error'))
                logger.info(f"    {status} {target_date}: {reason}")

        logger.info(f"    Summary: {region_success} successful, {region_fail} failed")

    logger.info("=" * 80)
    logger.info("PROCESS_HISTORICAL_RBEAST.PY COMPLETED")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()