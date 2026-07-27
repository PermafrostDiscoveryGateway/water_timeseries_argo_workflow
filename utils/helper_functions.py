import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os
import glob
from typing import List, Optional, Dict, Any
import zarr
# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from dotenv import load_dotenv
import time
from loguru import logger
import geemap
import ee
import psutil
from water_timeseries.downloader import EarthEngineDownloader
from water_timeseries.utils.spatial import create_longitude_latitude_grid, filter_gdf_by_bbox
from water_timeseries.dataset import DWDataset
from water_timeseries.breakpoint import NRTBreakpoint, BeastBreakpoint
import datetime
from utils.region_boundaries import get_region_boundaries
import json

import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import gc
import resource
import tracemalloc



# Enable memory tracking
def enable_memory_tracking():
    try:
        tracemalloc.start()
    except:
        pass


def get_memory_usage():
    """Get current memory usage in GB"""
    try:
        # Get RSS memory usage
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Convert to GB (on Linux, ru_maxrss is in KB)
        return rss / (1024 * 1024)
    except:
        return 0


def log_memory_usage(stage: str):
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    mem_gb = mem_mb / 1024

    try:
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        logger.debug(f"[MEMORY] {stage}: {mem_mb:.2f} MB ({mem_gb:.2f} GB) | Max RSS: {rss_gb:.2f} GB")
    except:
        logger.debug(f"[MEMORY] {stage}: {mem_mb:.2f} MB ({mem_gb:.2f} GB)")

    if mem_gb > 10:
        logger.warning(f"High memory usage detected: {mem_gb:.2f} GB at stage: {stage}")


def get_file_size_gb(file_path: str) -> float:
    """Get file size in GB"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 ** 3)
    return 0


def verify_merged_netcdf(file_path, expected_id_count=None, expected_date_count=None):
    """
    Verify a merged NetCDF file is valid and has expected dimensions.

    Args:
        file_path: Path to the NetCDF file
        expected_id_count: Optional expected number of IDs
        expected_date_count: Optional expected number of dates

    Returns:
        dict: Verification results with 'success' and 'valid' keys
    """
    try:
        logger.info(f"Verifying merged NetCDF file: {file_path}")
        ds = xr.open_dataset(file_path)

        id_count = len(ds['id_geohash'])
        date_count = len(ds['date'])

        result = {
            'success': True,  # Added: success key for compatibility
            'valid': True,    # Keep: existing valid key
            'id_count': id_count,
            'date_count': date_count,
            'file_size_gb': get_file_size_gb(str(file_path)),
            'variables': list(ds.data_vars)
        }

        if expected_id_count is not None and id_count != expected_id_count:
            logger.warning(f"ID count mismatch: expected {expected_id_count}, got {id_count}")
            result['valid'] = False
            result['success'] = False  # Added: set success to False
            result['id_count_mismatch'] = True

        if expected_date_count is not None and date_count != expected_date_count:
            logger.warning(f"Date count mismatch: expected {expected_date_count}, got {date_count}")
            result['valid'] = False
            result['success'] = False  # Added: set success to False
            result['date_count_mismatch'] = True

        ds.close()
        logger.info(f"✅ File verified: {id_count} IDs, {date_count} dates, {result['file_size_gb']:.2f} GB")
        return result

    except Exception as e:
        logger.error(f"❌ Failed to verify NetCDF file: {e}")
        return {
            'success': False,  # Added: success key
            'valid': False,    # Keep: existing valid key
            'error': str(e)
        }

def compare_netcdf_files(
        file1_path: str,
        file2_path: str,
        sample_ids: int = 5,
        variables_to_check: List[str] = None,
        verbose: bool = True
):
    """
    Compare two NetCDF files in detail and print the results.

    Args:
        file1_path: Path to the first NetCDF file (e.g., original)
        file2_path: Path to the second NetCDF file (e.g., merged)
        sample_ids: Number of random IDs to sample for detailed comparison
        variables_to_check: List of variable names to check (if None, checks all)
        verbose: If True, prints detailed information

    Returns:
        dict: Comparison results
    """

    def print_section(title, char='='):
        """Print a section header"""
        print(f"\n{char * 80}")
        print(f"{title}")
        print(f"{char * 80}")

    print_section("NETCDF FILE COMPARISON")
    print(f"File 1 (Original): {file1_path}")
    print(f"File 2 (Merged):    {file2_path}")
    print(f"Sample IDs to check: {sample_ids}")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Open both files
    logger.info(f"Opening {file1_path}...")
    ds1 = xr.open_dataset(file1_path)
    logger.info(f"Opening {file2_path}...")
    ds2 = xr.open_dataset(file2_path)

    results = {
        'file1': file1_path,
        'file2': file2_path,
        'dimensions': {},
        'variables': {},
        'sample_comparisons': [],
        'summary': {}
    }

    # ========== 1. BASIC FILE INFORMATION ==========
    print_section("1. BASIC FILE INFORMATION")

    # File sizes
    size1 = get_file_size_gb(file1_path)
    size2 = get_file_size_gb(file2_path)
    print(f"File 1 size: {size1:.4f} GB")
    print(f"File 2 size: {size2:.4f} GB")
    if size1 > 0:
        print(f"Size change: {((size2 - size1) / size1) * 100:+.2f}%")
    results['summary']['file1_size_gb'] = size1
    results['summary']['file2_size_gb'] = size2

    # ========== 2. DIMENSIONS ==========
    print_section("2. DIMENSIONS")

    for dim in ds1.dims:
        dim1_len = len(ds1[dim])
        dim2_len = len(ds2[dim]) if dim in ds2.dims else None

        print(f"Dimension '{dim}':")
        print(f"  File 1: {dim1_len}")
        print(f"  File 2: {dim2_len if dim2_len is not None else 'MISSING'}")

        if dim2_len is not None and dim1_len != dim2_len:
            diff = dim2_len - dim1_len
            print(f"  Difference: {diff:+d}")
            if dim == 'date':
                # Show which dates were added
                dates1 = set(pd.to_datetime(ds1[dim].values))
                dates2 = set(pd.to_datetime(ds2[dim].values))
                added_dates = dates2 - dates1
                if added_dates:
                    print(f"  Added dates ({len(added_dates)}):")
                    for date in sorted(added_dates):
                        print(f"    - {date.strftime('%Y-%m-%d')}")
            elif dim == 'id_geohash':
                # Show ID count change
                ids1 = set(ds1[dim].values)
                ids2 = set(ds2[dim].values)
                added_ids = ids2 - ids1
                removed_ids = ids1 - ids2
                if added_ids:
                    print(f"  Added IDs: {len(added_ids)}")
                    # Show a few sample added IDs
                    sample_added = list(added_ids)[:5]
                    print(f"    Sample added IDs: {sample_added}")
                if removed_ids:
                    print(f"  Removed IDs: {len(removed_ids)}")
                    sample_removed = list(removed_ids)[:5]
                    print(f"    Sample removed IDs: {sample_removed}")

        results['dimensions'][dim] = {
            'file1': dim1_len,
            'file2': dim2_len,
            'diff': dim2_len - dim1_len if dim2_len is not None else None
        }

    # ========== 3. VARIABLES ==========
    print_section("3. VARIABLES")

    # Check which variables exist in both files
    vars1 = set(ds1.data_vars)
    vars2 = set(ds2.data_vars)

    common_vars = vars1 & vars2
    only_in_1 = vars1 - vars2
    only_in_2 = vars2 - vars1

    print(f"Variables in File 1: {len(vars1)}")
    print(f"Variables in File 2: {len(vars2)}")
    print(f"Common variables: {len(common_vars)}")

    if only_in_1:
        print(f"Only in File 1: {sorted(only_in_1)}")
    if only_in_2:
        print(f"Only in File 2: {sorted(only_in_2)}")

    # Check variable properties
    print("\nVariable details:")
    for var in sorted(common_vars):
        print(f"\n  Variable '{var}':")

        # Data types
        dtype1 = ds1[var].dtype
        dtype2 = ds2[var].dtype
        print(f"    dtype: {dtype1} (File 1) vs {dtype2} (File 2)")
        if dtype1 != dtype2:
            print(f"    ⚠️  Data type mismatch!")

        # Shapes
        shape1 = ds1[var].shape
        shape2 = ds2[var].shape
        print(f"    shape: {shape1} (File 1) vs {shape2} (File 2)")
        if shape1 != shape2:
            print(f"    ⚠️  Shape mismatch!")

        # Encoding/compression
        encoding1 = ds1[var].encoding
        encoding2 = ds2[var].encoding

        has_zlib1 = encoding1.get('zlib', False)
        has_zlib2 = encoding2.get('zlib', False)
        complevel1 = encoding1.get('complevel', 0)
        complevel2 = encoding2.get('complevel', 0)
        shuffle1 = encoding1.get('shuffle', False)
        shuffle2 = encoding2.get('shuffle', False)
        chunksizes1 = encoding1.get('chunksizes', None)
        chunksizes2 = encoding2.get('chunksizes', None)

        print(f"    compression: zlib={has_zlib1}, level={complevel1} (File 1)")
        print(f"                zlib={has_zlib2}, level={complevel2} (File 2)")

        if has_zlib1 != has_zlib2 or complevel1 != complevel2:
            print(f"    ⚠️  Compression settings differ!")

        # Chunksizes
        if chunksizes1 and chunksizes2:
            print(f"    chunksizes: {chunksizes1} (File 1) vs {chunksizes2} (File 2)")
        elif chunksizes1:
            print(f"    chunksizes: {chunksizes1} (File 1) vs None (File 2)")
        elif chunksizes2:
            print(f"    chunksizes: None (File 1) vs {chunksizes2} (File 2)")

        # NaN counts (checking a few values)
        try:
            data1 = ds1[var].values
            data2 = ds2[var].values
            nan1 = np.isnan(data1).sum() if np.issubdtype(data1.dtype, np.number) else 0
            nan2 = np.isnan(data2).sum() if np.issubdtype(data2.dtype, np.number) else 0

            # Only show for numeric data
            if np.issubdtype(data1.dtype, np.number):
                min1, max1 = np.nanmin(data1), np.nanmax(data1)
                min2, max2 = np.nanmin(data2), np.nanmax(data2)
                mean1, std1 = np.nanmean(data1), np.nanstd(data1)
                mean2, std2 = np.nanmean(data2), np.nanstd(data2)

                print(f"    stats (File 1): min={min1:.4f}, max={max1:.4f}, mean={mean1:.4f}, std={std1:.4f}")
                print(f"    stats (File 2): min={min2:.4f}, max={max2:.4f}, mean={mean2:.4f}, std={std2:.4f}")
                print(f"    NaN count: {nan1} (File 1) vs {nan2} (File 2)")

                # Check if stats are similar
                if abs(mean1 - mean2) > 0.01 * abs(mean1):
                    print(f"    ⚠️  Mean values differ significantly!")
        except Exception as e:
            print(f"    Could not compute stats: {e}")

        results['variables'][var] = {
            'dtype': {'file1': str(dtype1), 'file2': str(dtype2)},
            'shape': {'file1': shape1, 'file2': shape2},
            'encoding': {'file1': encoding1, 'file2': encoding2}
        }

    # ========== 4. SAMPLE ID COMPARISON ==========
    print_section(f"4. SAMPLE ID COMPARISON (sampling {sample_ids} random IDs)")

    # Get common IDs
    ids1 = set(ds1['id_geohash'].values)
    ids2 = set(ds2['id_geohash'].values)
    common_ids = ids1 & ids2

    if not common_ids:
        print("⚠️  No common IDs found between the two files!")
    else:
        # Sample random IDs
        sample_id_list = list(np.random.choice(list(common_ids), min(sample_ids, len(common_ids)), replace=False))

        for sample_id in sample_id_list:
            print(f"\n  ID: {sample_id}")
            results['sample_comparisons'].append({'id': sample_id})

            # Get data for this ID
            data1 = ds1.sel(id_geohash=sample_id)
            data2 = ds2.sel(id_geohash=sample_id)

            # Check dimensions
            print(f"    dates: {len(data1['date'])} (File 1) vs {len(data2['date'])} (File 2)")

            # Check if dates match
            dates1 = set(pd.to_datetime(data1['date'].values))
            dates2 = set(pd.to_datetime(data2['date'].values))
            common_dates = dates1 & dates2
            added_dates = dates2 - dates1

            print(f"    common dates: {len(common_dates)}")
            if added_dates:
                print(f"    added dates: {len(added_dates)}")
                for date in sorted(added_dates)[:3]:
                    print(f"      - {date.strftime('%Y-%m-%d')}")

            # Check variable values
            all_match = True
            for var in common_vars:
                if var in data1 and var in data2:
                    try:
                        # Get values for common dates
                        if len(common_dates) > 0:
                            common_dates_list = sorted(common_dates)
                            vals1 = data1[var].sel(date=pd.to_datetime(list(common_dates_list))).values
                            vals2 = data2[var].sel(date=pd.to_datetime(list(common_dates_list))).values

                            # Check if values match (allow small floating point differences)
                            if np.issubdtype(vals1.dtype, np.number):
                                if np.allclose(vals1, vals2, rtol=1e-6, atol=1e-6):
                                    print(f"    {var}: ✅ matches on common dates")
                                else:
                                    max_diff = np.max(np.abs(vals1 - vals2))
                                    print(f"    {var}: ⚠️  DIFFERS (max diff: {max_diff:.6f})")
                                    all_match = False
                            else:
                                # For non-numeric data, check exact equality
                                if np.array_equal(vals1, vals2):
                                    print(f"    {var}: ✅ matches on common dates")
                                else:
                                    print(f"    {var}: ⚠️  DIFFERS")
                                    all_match = False
                    except Exception as e:
                        print(f"    {var}: Could not compare - {e}")

            results['sample_comparisons'][-1]['all_match'] = all_match

    # ========== 5. NEW DATA VERIFICATION ==========
    print_section("5. NEW DATA VERIFICATION")

    # Check if new IDs were added
    if len(ids2) > len(ids1):
        new_ids = ids2 - ids1
        print(f"✅ File 2 has {len(new_ids)} new IDs that weren't in File 1")
        if len(new_ids) <= 10:
            print(f"   New IDs: {sorted(new_ids)}")
        else:
            print(f"   Sample of new IDs: {list(new_ids)[:10]}...")

        # Verify that new IDs have data for the expected dates
        for new_id in list(new_ids)[:min(3, len(new_ids))]:
            new_data = ds2.sel(id_geohash=new_id)
            date_count = len(new_data['date'])
            print(f"   ID {new_id}: has {date_count} dates")

        results['summary']['new_ids_added'] = len(new_ids)
    elif len(ids2) == len(ids1):
        print("File 2 has the same number of IDs as File 1")
        # But check if dates were added
        dates1 = set(pd.to_datetime(ds1['date'].values))
        dates2 = set(pd.to_datetime(ds2['date'].values))
        if len(dates2) > len(dates1):
            added_dates = dates2 - dates1
            print(f"✅ New dates added: {len(added_dates)}")
            for date in sorted(added_dates)[:5]:
                print(f"   - {date.strftime('%Y-%m-%d')}")
            results['summary']['new_dates_added'] = len(added_dates)
        else:
            print("No new IDs or dates were added - files may be identical")
    else:
        removed_ids = ids1 - ids2
        print(f"⚠️  File 2 has FEWER IDs than File 1 ({len(removed_ids)} removed)")
        results['summary']['ids_removed'] = len(removed_ids)

    # ========== 6. FINAL SUMMARY ==========
    print_section("6. SUMMARY")

    # Determine if the merge was successful
    is_successful = True
    issues = []

    # Check dimensions
    for dim, dim_info in results['dimensions'].items():
        if dim_info['file2'] is None:
            is_successful = False
            issues.append(f"Dimension '{dim}' missing in File 2")
        elif dim_info['file1'] != dim_info['file2']:
            if dim == 'date':
                # Date dimension increasing is expected
                if dim_info['file2'] > dim_info['file1']:
                    print(f"✅ Date dimension increased by {dim_info['diff']} (expected for NRT update)")
                else:
                    is_successful = False
                    issues.append(f"Date dimension decreased unexpectedly")
            elif dim == 'id_geohash':
                # ID dimension changing is expected
                if dim_info['file2'] >= dim_info['file1']:
                    print(f"✅ ID dimension has {dim_info['diff']} more IDs (expected)")
                else:
                    is_successful = False
                    issues.append(f"ID dimension decreased unexpectedly")
            else:
                is_successful = False
                issues.append(f"Dimension '{dim}' changed from {dim_info['file1']} to {dim_info['file2']}")

    # Check if original IDs were preserved
    missing_original_ids = ids1 - ids2
    if missing_original_ids:
        print(f"⚠️  {len(missing_original_ids)} original IDs are missing from the new file!")
        is_successful = False
        issues.append(f"Missing original IDs: {len(missing_original_ids)}")
    else:
        print("✅ All original IDs are preserved in the new file")

    # Check if any variables were lost
    missing_vars = vars1 - vars2
    if missing_vars:
        print(f"⚠️  Variables missing in File 2: {sorted(missing_vars)}")
        is_successful = False
        issues.append(f"Missing variables: {sorted(missing_vars)}")
    else:
        print("✅ All variables are preserved in the new file")

    # Check compression
    compression_changed = False
    for var in common_vars:
        enc1 = ds1[var].encoding
        enc2 = ds2[var].encoding
        if enc1.get('zlib', False) != enc2.get('zlib', False):
            compression_changed = True
            break
    if compression_changed:
        print("✅ Compression settings were updated (improved)")
    else:
        print("ℹ️  Compression settings unchanged")

    # Final verdict
    print_section("FINAL VERDICT", '=')
    if is_successful:
        print("✅ MERGE SUCCESSFUL - All checks passed!")
        print(f"   - Original file: {size1:.2f} GB")
        print(f"   - New file: {size2:.2f} GB")
        print(f"   - Size change: {((size2 - size1) / size1) * 100:+.1f}%")

        # Calculate compression ratio
        if size1 > 0:
            compression_ratio = size1 / size2
            if compression_ratio > 1:
                print(f"   - Compression ratio: {compression_ratio:.2f}x (new file is smaller)")
            else:
                print(f"   - Compression ratio: {compression_ratio:.2f}x (new file is larger)")
    else:
        print("❌ MERGE HAS ISSUES:")
        for issue in issues:
            print(f"   - {issue}")

    results['summary']['successful'] = is_successful
    results['summary']['issues'] = issues

    # Close datasets
    ds1.close()
    ds2.close()

    return results


def generate_expected_dates(
        start_year: int = 2015,
        end_year: int = None,
        end_month: int = None,
        months: List[int] = [6, 7, 8, 9]  # June, July, August, September
):
    """
    Generate a list of expected dates for specific months (June-September by default).

    Args:
        start_year: Year to start from (default: 2015)
        end_year: Year to end at (default: current year)
        end_month: Month to end at (default: current month)
        months: List of months to include (default: [6, 7, 8, 9])

    Returns:
        List of pandas Timestamps for the first of each month in the specified months
    """
    if end_year is None:
        end_year = datetime.datetime.now().year
    if end_month is None:
        end_month = datetime.datetime.now().month

    dates = []

    for year in range(start_year, end_year + 1):
        for month in months:
            # Skip if year is end_year and month is after end_month
            if year == end_year and month > end_month:
                continue
            # Create date for first of the month
            dates.append(pd.Timestamp(f"{year}-{month:02d}-01"))

    return dates


def download_near_real_time_region_dates(
        region: str = "TEST",
        run_start_label: str = None,
        env_path: str = None,
        dates_to_download: List[pd.Timestamp] = None
):
    """
    Download near-real-time data for a specific region for specified dates.

    This function ONLY handles downloading - no merging is performed.
    Downloads are saved to the download directory for later merging.

    Downloads are skipped only if the file already exists AND has content (> 0 bytes).
    Empty or corrupted files will be overwritten.

    Args:
        region: Region name (e.g., "TEST", "AFRICA", "SOUTH_AMERICA")
        run_start_label: Optional label for tracking runs
        env_path: Optional path to .env file
        dates_to_download: List of pandas Timestamps to download. If None,
                          generates expected dates from 2015 to present for months 6-9.

    Returns:
        dict: Status information including success/failure counts per date
    """
    log_memory_usage("Download function start")

    region_boundaries = get_region_boundaries()

    start = datetime.datetime.now()
    logger.debug(f"Current time: {datetime.datetime.now()}")

    # Load environment variables
    if env_path:
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    REGION_NAME = region

    output_dir = os.environ['output_dir']
    output_dir = os.path.join(output_dir, REGION_NAME)
    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID

    try:
        ee.Initialize(project=EE_PROJECT_ID)
        logger.debug("Earth engine successfully initialized")
    except Exception as e:
        logger.debug(f"Failed to initialize earth engine: {e}")

    try:
        geemap.ee_initialize(project=EE_PROJECT_ID)
        logger.debug("Initialized geemap")
    except Exception as e:
        logger.debug(f"Failed to initialize geemap: {e}")

    dynamic_world_data_dir = os.environ['dynamic_world_data']
    dynamic_world_download_dir = Path(os.environ['dynamic_world_downloads'])
    dynamic_world_download_dir.mkdir(exist_ok=True, parents=True)

    logger.debug(f"Region name is {REGION_NAME}")

    bounding_box_coords = region_boundaries[REGION_NAME]

    logger.debug(f"Bounding box coordinates are {bounding_box_coords}")
    time.sleep(15)

    X_MIN_START = bounding_box_coords['X_MIN_START']
    X_MIN_END = bounding_box_coords['X_MIN_END']
    Y_MIN_START = bounding_box_coords['Y_MIN_START']
    Y_MIN_END = bounding_box_coords['Y_MIN_END']

    # ========== DETERMINE WHICH DATES TO DOWNLOAD ==========
    if dates_to_download is not None:
        # Use explicitly provided dates
        dates_to_download = dates_to_download
        logger.info(f"Processing {len(dates_to_download)} explicitly provided dates")
        for d in dates_to_download:
            logger.info(f"  - {d.strftime('%Y-%m-%d')}")
    else:
        # Generate expected dates from 2015 to present for June-September
        dates_to_download = generate_expected_dates(start_year=2015)
        logger.info(f"Generated {len(dates_to_download)} expected dates from 2015 to present")
        logger.info("Months included: June, July, August, September")

    if not dates_to_download:
        logger.info("No dates to process")
        return {'success': True, 'dates_processed': [], 'message': 'No dates to process'}

    vector_lake_file = os.environ['vector_lake_file']
    path_lake_vector = vector_lake_file

    # Track results for all dates
    all_results = {}
    overall_success = True

    # ========== LOAD ORIGINAL VALID IDs ONCE ==========
    # Get the most recent historical file to get valid IDs
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
    if not all_dynamic_world_files:
        logger.error(f"No .nc files found in {dynamic_world_data_dir}")
        return {'success': False, 'error': 'No .nc files found'}

    original_historical_file = max(all_dynamic_world_files, key=os.path.getctime)
    logger.info("Loading original historical dataset to get valid IDs...")
    ds_original = xr.open_dataset(original_historical_file)
    original_valid_ids = set(ds_original['id_geohash'].values)
    ds_original.close()
    logger.info(f"Found {len(original_valid_ids)} valid IDs in original historical dataset")

    # Load GDF once for the region (same for all dates)
    gdf = gpd.read_parquet(path_lake_vector)
    log_memory_usage("After loading lake vectors")

    # Initialize downloader once
    downloader = EarthEngineDownloader(ee_project=EE_PROJECT_ID)

    # Process each date to download
    for date_idx, date in enumerate(dates_to_download):
        ANALYSIS_DATE = date.strftime("%Y-%m")
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing date {date_idx + 1}/{len(dates_to_download)}: {ANALYSIS_DATE}")
        logger.info(f"{'=' * 80}")

        date_start = datetime.datetime.now()

        # Track results for this date
        date_results = {
            'analysis_date': ANALYSIS_DATE,
            'success_bbox_downloads': 0,
            'failed_bbox_downloads': 0,
            'skipped_bbox_downloads': 0,
            'overwritten_bbox_downloads': 0,
            'expected_downloads': 0,
            'grid_tiles_processed': [],
            'successful': False
        }

        bbox_size_lon = 1
        bbox_size_lat = 1
        grid = create_longitude_latitude_grid(
            lon_range=(X_MIN_START, X_MIN_END),
            lat_range=(Y_MIN_START, Y_MIN_END),
            bbox_size_lon=bbox_size_lon,
            bbox_size_lat=bbox_size_lat
        )
        logger.info(f'Created grid with {len(grid)} tiles')
        log_memory_usage("After creating grid")

        current_download_dir = Path(str(dynamic_world_download_dir), REGION_NAME, f'download_{ANALYSIS_DATE}')
        current_download_dir.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Current download directory: {current_download_dir}")

        if not hasattr(geemap, 'ee_initialize'):
            logger.warning("geemap.ee_initialize missing, adding runtime patch")

            def ee_initialize(project=None, **kwargs):
                if project:
                    ee.Initialize(project=project, **kwargs)
                else:
                    ee.Initialize(**kwargs)

            geemap.ee_initialize = ee_initialize
            logger.info("Runtime patch applied to geemap")

        # Create run label for this date
        if run_start_label is None:
            date_run_label = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        else:
            date_run_label = f"{run_start_label}_{ANALYSIS_DATE}"

        # Files for tracking downloads
        outfile_downloads_failed_file = current_download_dir / f'grid_tiles_download_failed_{date_run_label}.txt'
        outfile_downloads_success_file = current_download_dir / f'grid_tiles_download_success_{date_run_label}.txt'
        outfile_downloads_overwritten_file = current_download_dir / f'grid_tiles_download_overwritten_{date_run_label}.txt'

        # Track expected grid tiles
        expected_grid_tiles = []

        total = len(grid[:])
        logger.debug(f"There are total {total} grid tiles for {REGION_NAME}")

        for i, (lon, lat) in enumerate(tqdm(grid[:], total=total, desc=f"Downloading {ANALYSIS_DATE}")):
            logger.debug(f"Processing {i}/{total} grid tiles.")
            bbox_west = int(lon)
            bbox_east = int(lon + bbox_size_lon)
            bbox_south = int(lat)
            bbox_north = int(lat + bbox_size_lat)

            grid_coords = f"{bbox_west}_{bbox_east}_{bbox_south}_{bbox_north}"
            print(f"Processing download for bbox: {bbox_west} {bbox_east} {bbox_south} {bbox_north}")

            outfile_download = current_download_dir / f'DW_{ANALYSIS_DATE}_{bbox_west}_{bbox_east}_{bbox_south}_{bbox_north}.nc'

            gdf_subset = filter_gdf_by_bbox(
                gdf=gdf,
                bbox_west=lon,
                bbox_east=lon + bbox_size_lon,
                bbox_south=lat,
                bbox_north=lat + bbox_size_lat
            )
            n_lakes = len(gdf_subset)
            print('Number of lakes: ', n_lakes)

            id_list = gdf_subset['id_geohash'].values.tolist()
            if n_lakes == 0:
                print(f'No lakes for grid {bbox_west} {bbox_south}. Skipping!')
                continue

            # ========== USE ORIGINAL VALID IDs ==========
            original_count = len(id_list)
            id_list = [id_val for id_val in id_list if id_val in original_valid_ids]
            filtered_count = len(id_list)

            if filtered_count == 0:
                print(
                    f'WARNING: No valid historical IDs for grid {bbox_west} {bbox_south} (had {original_count} lakes, none in historical data). Skipping!')
                continue
            elif filtered_count < original_count:
                print(
                    f'NOTE: Filtered {original_count - filtered_count} lakes not found in historical data. Processing {filtered_count} lakes.')
                gdf_subset = gdf_subset[gdf_subset['id_geohash'].isin(id_list)]

            # This grid tile should be processed
            expected_grid_tiles.append(grid_coords)
            date_results['expected_downloads'] += 1

            # ========== CHECK IF FILE EXISTS AND HAS CONTENT ==========
            should_download = True
            should_overwrite = False

            if outfile_download.exists():
                # Check if file has content (not empty)
                file_size = outfile_download.stat().st_size
                if file_size > 0:
                    # File exists and has content - skip download
                    print(
                        f'Download already exists with content ({file_size} bytes) for {bbox_west} {bbox_south}! Skipping download.')
                    date_results['skipped_bbox_downloads'] += 1
                    with open(outfile_downloads_success_file, 'a') as f:
                        f.write(f"{ANALYSIS_DATE}_{grid_coords}\n")
                    date_results['grid_tiles_processed'].append(grid_coords)
                    should_download = False
                else:
                    # File exists but is empty - overwrite it
                    print(f'Download file exists but is empty (0 bytes) for {bbox_west} {bbox_south}. Will overwrite.')
                    date_results['overwritten_bbox_downloads'] += 1
                    should_overwrite = True
                    # Remove the empty file before downloading
                    try:
                        outfile_download.unlink()
                        print(f'Removed empty file: {outfile_download}')
                    except Exception as e:
                        logger.warning(f"Could not remove empty file {outfile_download}: {e}")

            if not should_download:
                continue

            # Download data (either new or overwriting)
            download_successful = False
            try:
                n_features = len(gdf_subset)
                if n_features > 500:
                    max_total_requests = min(100, n_features)
                    logger.debug(f"Grid has {n_features} features, using max_requests={max_total_requests}")
                else:
                    max_total_requests = 500

                ds_dl = downloader.download_dw_monthly(
                    gdf=gdf_subset,
                    max_total_requests=max_total_requests,
                    n_parallel=2,
                    date_list=[ANALYSIS_DATE],
                    save_to_file=outfile_download
                )

                if ds_dl is not None:
                    download_successful = True
                    if should_overwrite:
                        print(f'Successfully re-downloaded data for {bbox_west} {bbox_south} (overwrote empty file)')
                        with open(outfile_downloads_overwritten_file, 'a') as f:
                            f.write(f"{ANALYSIS_DATE}_{grid_coords}\n")
                    else:
                        print(f'Successfully downloaded data for {bbox_west} {bbox_south}')
                    date_results['success_bbox_downloads'] += 1
                    with open(outfile_downloads_success_file, 'a') as f:
                        f.write(f"{ANALYSIS_DATE}_{grid_coords}\n")
                    date_results['grid_tiles_processed'].append(grid_coords)
                else:
                    print(f'WARNING: No data available for {bbox_west} {bbox_south} on {ANALYSIS_DATE}')
                    date_results['failed_bbox_downloads'] += 1
                    with open(outfile_downloads_failed_file, 'a') as f:
                        f.write(f"{ANALYSIS_DATE}_{grid_coords}\n")

            except ValueError as e:
                if "No data was extracted" in str(e):
                    print(f'WARNING: No data available for {bbox_west} {bbox_south} on {ANALYSIS_DATE}')
                else:
                    logger.error(f"Download error for {bbox_west} {bbox_south}: {e}")
                date_results['failed_bbox_downloads'] += 1
                with open(outfile_downloads_failed_file, 'a') as f:
                    f.write(f"{ANALYSIS_DATE}_{grid_coords}\n")
            except Exception as e:
                logger.error(f"Unexpected error downloading {bbox_west} {bbox_south}: {e}")
                date_results['failed_bbox_downloads'] += 1
                with open(outfile_downloads_failed_file, 'a') as f:
                    f.write(f"{ANALYSIS_DATE}_{grid_coords}\n")

            # Clean up
            if 'ds_dl' in locals() and ds_dl is not None:
                ds_dl.close()
                del ds_dl
                gc.collect()

        # ========== CREATE MANIFEST FILE FOR THIS DATE ==========
        manifest_file = current_download_dir / f'download_manifest_{date_run_label}.json'
        manifest_data = {
            'region': REGION_NAME,
            'analysis_date': ANALYSIS_DATE,
            'run_start_label': date_run_label,
            'expected_downloads': date_results['expected_downloads'],
            'successful_downloads': date_results['success_bbox_downloads'],
            'failed_downloads': date_results['failed_bbox_downloads'],
            'skipped_downloads': date_results['skipped_bbox_downloads'],
            'overwritten_downloads': date_results['overwritten_bbox_downloads'],
            'expected_grid_tiles': expected_grid_tiles,
            'timestamp': datetime.datetime.now().isoformat()
        }
        with open(manifest_file, 'w') as f:
            json.dump(manifest_data, f, indent=2)

        # Determine if this date's downloads were successful
        date_results['successful'] = (date_results['failed_bbox_downloads'] == 0 and
                                      date_results['expected_downloads'] > 0)

        # Create completion marker
        if date_results['successful']:
            completion_file = current_download_dir / f'download_complete_{date_run_label}.success'
            with open(completion_file, 'w') as f:
                f.write(f"All {date_results['expected_downloads']} downloads completed successfully\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            logger.info(f"✅ All downloads completed successfully for {ANALYSIS_DATE}")
        else:
            completion_file = current_download_dir / f'download_complete_{date_run_label}.partial'
            with open(completion_file, 'w') as f:
                f.write(
                    f"Downloads completed with {date_results['failed_bbox_downloads']} failures out of {date_results['expected_downloads']}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            logger.warning(
                f"⚠️ Downloads completed with {date_results['failed_bbox_downloads']} failures for {ANALYSIS_DATE}")
            overall_success = False

        date_end = datetime.datetime.now()
        logger.debug(f"Finished download for date {ANALYSIS_DATE} in {date_end - date_start}")
        logger.info(f"Downloads for {ANALYSIS_DATE}: {date_results['success_bbox_downloads']} successful, "
                    f"{date_results['failed_bbox_downloads']} failed, "
                    f"{date_results['skipped_bbox_downloads']} skipped, "
                    f"{date_results['overwritten_bbox_downloads']} overwritten")

        # Store results for this date
        all_results[ANALYSIS_DATE] = date_results

    # ========== SUMMARY ==========
    logger.info(f"\n{'=' * 80}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'=' * 80}")

    # Count successful dates
    successful_dates = sum(1 for r in all_results.values() if r['successful'])
    failed_dates = sum(1 for r in all_results.values() if not r['successful'])

    for date, results in all_results.items():
        status = "✅ SUCCESS" if results['successful'] else "⚠️ PARTIAL"
        logger.info(f"{date}: {status} - {results['success_bbox_downloads']} successful, "
                    f"{results['failed_bbox_downloads']} failed, "
                    f"{results['skipped_bbox_downloads']} skipped, "
                    f"{results['overwritten_bbox_downloads']} overwritten")

    logger.info(f"Overall status: {'✅ SUCCESS' if overall_success else '⚠️ PARTIAL FAILURE'}")
    logger.info(f"Successfully downloaded {successful_dates}/{len(dates_to_download)} dates")

    return {
        'success': overall_success,
        'dates_processed': list(all_results.keys()),
        'date_results': all_results,
        'total_dates': len(all_results),
        'successful_dates': successful_dates,
        'failed_dates': failed_dates,
        'dates_to_download': [d.strftime("%Y-%m") for d in dates_to_download]
    }

# ========== NEW: MERGE FUNCTION ==========
def get_ids_for_region_from_vector_file(region: str, env_path: str = None) -> List[str]:
    """
    Get IDs for a region from the vector lake file.
    Handles both Point and Polygon geometries.
    """
    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Get region boundaries
    from utils.region_boundaries import get_region_boundaries
    region_boundaries = get_region_boundaries()

    if region not in region_boundaries:
        logger.error(f"Region {region} not found in boundaries")
        return []

    bounds = region_boundaries[region]
    x_min_start = bounds['X_MIN_START']
    x_min_end = bounds['X_MIN_END']
    y_min_start = bounds['Y_MIN_START']
    y_min_end = bounds['Y_MIN_END']

    # Load the vector lake file
    vector_lake_file = os.environ.get('vector_lake_file')
    if not vector_lake_file:
        logger.error("vector_lake_file not set in environment")
        return []

    if not Path(vector_lake_file).exists():
        logger.error(f"Vector lake file not found: {vector_lake_file}")
        return []

    try:
        # Load GDF
        gdf = gpd.read_parquet(vector_lake_file)

        # Get geometry type
        geom_type = gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else None

        # Get coordinates based on geometry type
        if geom_type in ['Polygon', 'MultiPolygon']:
            # Use centroids for polygon geometries
            centroids = gdf.geometry.centroid
            x_coords = centroids.x
            y_coords = centroids.y
        elif geom_type == 'Point':
            # Use point coordinates directly
            x_coords = gdf.geometry.x
            y_coords = gdf.geometry.y
        else:
            # Fallback - try representative point
            logger.warning(f"Unsupported geometry type: {geom_type}, using representative_point()")
            rep_points = gdf.geometry.representative_point()
            x_coords = rep_points.x
            y_coords = rep_points.y

        # Filter by bounding box
        mask = (x_coords >= x_min_start) & (x_coords <= x_min_end) & \
               (y_coords >= y_min_start) & (y_coords <= y_min_end)

        gdf_subset = gdf[mask]

        # Get the IDs
        if 'id_geohash' in gdf_subset.columns:
            ids = gdf_subset['id_geohash'].values.tolist()
        else:
            # Try alternative column names
            id_column = None
            for col in gdf_subset.columns:
                if 'id' in col.lower() or 'geohash' in col.lower():
                    id_column = col
                    break

            if id_column:
                ids = gdf_subset[id_column].values.tolist()
                logger.info(f"Using column '{id_column}' for IDs")
            else:
                logger.error("No ID column found in vector file")
                return []

        logger.info(f"Found {len(ids)} IDs for region {region} from vector file")
        return ids

    except Exception as e:
        logger.error(f"Error getting region IDs from vector file: {e}")
        import traceback
        traceback.print_exc()
        return []

def merge_new_results(
        region: str = 'TEST',
        date_to_merge: str = None,
        merged_file_path: str = None,
        env_path: str = None
) -> Dict[str, Any]:
    """
    Merge new downloaded results for a single date into a NetCDF file.

    This function:
    1. Finds all downloaded files for the specified date
    2. Combines them into a single dataset
    3. Saves the combined data to the specified file path

    Args:
        region: Region name (e.g., "TEST", "AFRICA")
        date_to_merge: Date in "YYYY-MM" format
        merged_file_path: Path where the merged file should be saved
        env_path: Optional path to .env file

    Returns:
        dict: Result with status, file path, and statistics
    """
    logger.debug(f"Merging new results for {region} and {date_to_merge} into file {merged_file_path}")

    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Validate inputs
    if date_to_merge is None:
        logger.error("date_to_merge is required")
        return {'success': False, 'error': 'date_to_merge is required'}

    if merged_file_path is None:
        dynamic_world_data_dir = os.environ.get('dynamic_world_data')
        if not dynamic_world_data_dir:
            logger.error("dynamic_world_data not set in environment")
            return {'success': False, 'error': 'dynamic_world_data not set'}
        merged_file_path = os.path.join(dynamic_world_data_dir, f"dw_{region}_{date_to_merge}.nc")

    # Ensure directory exists
    Path(merged_file_path).parent.mkdir(parents=True, exist_ok=True)

    # Find downloaded files for this date
    dynamic_world_download_dir = Path(os.environ.get('dynamic_world_downloads', ''))
    download_dir = dynamic_world_download_dir / region / f'download_{date_to_merge}'

    if not download_dir.exists():
        logger.warning(f"Download directory does not exist: {download_dir}")
        return {'success': False, 'error': f'Download directory not found: {download_dir}'}

    # Get all downloaded NetCDF files for this date
    downloaded_files = sorted(glob.glob(str(download_dir / f'DW_{date_to_merge}_*.nc')))

    if not downloaded_files:
        logger.warning(f"No downloaded files found for {date_to_merge} in {download_dir}")
        return {'success': False, 'error': f'No downloaded files found for {date_to_merge}'}

    logger.info(f"Found {len(downloaded_files)} downloaded files for {date_to_merge}")

    try:
        # Combine all downloaded files into a single dataset
        logger.info("Combining downloaded files...")
        combined = None
        failed_files = []

        for nc_file in downloaded_files:
            try:
                ds = xr.open_dataset(nc_file)
                if len(ds['id_geohash']) > 0:
                    if combined is None:
                        combined = ds
                    else:
                        # Concatenate along id_geohash dimension
                        combined = xr.concat([combined, ds], dim='id_geohash')
                        # Remove duplicate IDs
                        _, unique_idx = np.unique(combined['id_geohash'].values, return_index=True)
                        if len(unique_idx) < len(combined['id_geohash']):
                            combined = combined.isel(id_geohash=np.sort(unique_idx))
                else:
                    logger.warning(f"File {nc_file} has no IDs, skipping")
                    failed_files.append(nc_file)
            except Exception as e:
                logger.error(f"Error opening {nc_file}: {e}")
                failed_files.append(nc_file)

        if combined is None:
            logger.error("No valid data to merge")
            return {'success': False, 'error': 'No valid data to merge'}

        logger.info(f"Combined dataset has {len(combined['id_geohash'])} IDs and {len(combined['date'])} dates")

        # Ensure date dimension is correct
        if len(combined['date']) > 0:
            # Sort by date
            combined = combined.sortby('date')

        # Write to NetCDF with compression
        logger.info(f"Writing merged data to {merged_file_path}")

        encoding = {}
        for var in combined.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 4,
                'shuffle': True
            }

        # Write to file
        combined.to_netcdf(merged_file_path, encoding=encoding)

        # Get file size
        file_size_gb = Path(merged_file_path).stat().st_size / (1024 ** 3)

        # Clean up
        combined.close()
        gc.collect()

        result = {
            'success': True,
            'file_path': merged_file_path,
            'id_count': len(combined['id_geohash']),
            'date_count': len(combined['date']),
            'file_size_gb': round(file_size_gb, 4),
            'files_merged': len(downloaded_files) - len(failed_files),
            'files_failed': len(failed_files),
            'failed_files': failed_files if failed_files else None,
            'region': region,
            'date': date_to_merge
        }

        logger.info(f"✅ Merge completed successfully!")
        logger.info(f"  File: {merged_file_path}")
        logger.info(f"  IDs: {result['id_count']:,}")
        logger.info(f"  Dates: {result['date_count']}")
        logger.info(f"  Size: {result['file_size_gb']:.4f} GB")

        return result

    except Exception as e:
        logger.error(f"Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def has_region_been_merged_for_dates(
        region: str,
        dates_to_check: List[str],
        historical_file_path: str = None,
        env_path: str = None
) -> dict:
    """
    Check if a specific region already has the given dates merged.

    Uses the vector lake file to get region IDs (same method as download),
    then checks if those IDs have data in the NetCDF file.
    """
    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Find the historical file if not provided
    if historical_file_path is None:
        dynamic_world_data_dir = os.environ.get('dynamic_world_data')
        if not dynamic_world_data_dir:
            return {
                'all_dates_present': False,
                'some_dates_present': False,
                'none_dates_present': True,
                'error': 'dynamic_world_data not set in environment',
                'missing_dates': dates_to_check,
                'present_dates': [],
                'partial_dates': []
            }

        all_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
        if not all_files:
            return {
                'all_dates_present': False,
                'some_dates_present': False,
                'none_dates_present': True,
                'error': 'No NetCDF files found',
                'missing_dates': dates_to_check,
                'present_dates': [],
                'partial_dates': []
            }

        historical_file_path = max(all_files, key=lambda f: Path(f).stat().st_mtime)

    try:
        # Step 1: Get IDs for this region from the vector file
        region_ids = get_ids_for_region_from_vector_file(region, env_path)

        # Handle regions with NO IDs
        if not region_ids:
            logger.info(f"Region {region} has NO IDs (no lakes in this region)")
            return {
                'all_dates_present': True,
                'some_dates_present': False,
                'none_dates_present': False,
                'present_dates': dates_to_check,
                'missing_dates': [],
                'partial_dates': [],
                'region_id_count': 0,
                'region_ids_in_file': 0,
                'region_ids_missing': 0,
                'file_path': historical_file_path,
                'region_found': True,
                'region_has_ids': False,
                'no_data_expected': True,
                'message': f'Region {region} has no IDs - nothing to merge'
            }

        logger.info(f"Region {region} has {len(region_ids)} IDs to check")

        # Step 2: Open the NetCDF file
        logger.info(f"Opening NetCDF file: {historical_file_path}")
        ds = xr.open_dataset(historical_file_path)

        # Get all IDs that exist in the NetCDF file
        existing_ids = set(ds['id_geohash'].values)

        # Find which region IDs are in the NetCDF file
        region_ids_in_file = [id_val for id_val in region_ids if id_val in existing_ids]
        region_ids_missing = [id_val for id_val in region_ids if id_val not in existing_ids]

        logger.info(f"Region {region}: {len(region_ids_in_file)} IDs in file, {len(region_ids_missing)} IDs missing")

        # Handle case: No region IDs in file
        if not region_ids_in_file:
            ds.close()
            return {
                'all_dates_present': False,
                'some_dates_present': False,
                'none_dates_present': True,
                'present_dates': [],
                'missing_dates': dates_to_check,
                'partial_dates': [],
                'file_path': historical_file_path,
                'region_found': True,
                'region_has_ids': True,
                'region_id_count': len(region_ids),
                'region_ids_in_file': 0,
                'region_ids_missing': len(region_ids),
                'no_data_expected': False,
                'message': f'No IDs for region {region} found in NetCDF file'
            }

        # Step 3: Check dates for the region IDs that exist
        region_data = ds.sel(id_geohash=region_ids_in_file)
        existing_dates = set(pd.to_datetime(region_data['date'].values))
        existing_date_strings = {d.strftime("%Y-%m") for d in existing_dates}

        logger.info(f"Region {region} has {len(existing_date_strings)} dates in file")

        # ========== DETERMINE WHICH VARIABLE TO USE FOR DATA CHECK ==========
        # Based on inspection: the file has 'water' not 'water_observed'
        # Priority order for checking data presence
        data_vars = list(region_data.data_vars)
        logger.debug(f"Available variables: {data_vars}")

        var_candidates = ['water', 'water_observed', 'water_predicted', 'water_observed_qa']

        data_var = None
        for var in var_candidates:
            if var in region_data.data_vars:
                data_var = var
                logger.info(f"Using '{data_var}' to check data presence")
                break

        if data_var is None:
            # No suitable variable found - log warning and use date presence as proxy
            logger.warning(f"No data variable found in file. Available: {data_vars}")
            logger.warning("Using date presence as proxy for data availability")

            # Fallback: just check if dates exist in the file
            present_dates = []
            missing_dates = []
            partial_dates = []

            for date_str in dates_to_check:
                if date_str in existing_date_strings:
                    present_dates.append(date_str)
                else:
                    missing_dates.append(date_str)

            all_present = len(missing_dates) == 0 and len(partial_dates) == 0
            some_present = len(present_dates) > 0 or len(partial_dates) > 0
            none_present = len(present_dates) == 0 and len(partial_dates) == 0

            ds.close()

            result = {
                'all_dates_present': all_present,
                'some_dates_present': some_present,
                'none_dates_present': none_present,
                'present_dates': present_dates,
                'missing_dates': missing_dates,
                'partial_dates': partial_dates,
                'file_path': historical_file_path,
                'region_found': True,
                'region_has_ids': True,
                'region_id_count': len(region_ids),
                'region_ids_in_file': len(region_ids_in_file),
                'region_ids_missing': len(region_ids_missing),
                'no_data_expected': False,
                'data_var_used': 'date_presence_only',
                'warning': 'No water data variable found, used date presence as proxy'
            }

            logger.warning(f"Using fallback check for region {region}: date presence only")
            return result

        # Step 4: Check which dates are present using the data variable
        present_dates = []
        missing_dates = []
        partial_dates = []

        for date_str in dates_to_check:
            if date_str in existing_date_strings:
                # Check if ALL region IDs have data for this date
                try:
                    date_ts = pd.Timestamp(f"{date_str}-01")
                    date_data = region_data.sel(date=date_ts)

                    # Count IDs with non-NaN data for this date
                    if data_var in date_data.data_vars:
                        # Check if the data is all NaN (no data) or has values
                        data_values = date_data[data_var].values

                        # Check if any values are non-NaN
                        has_data_mask = ~np.isnan(data_values)
                        ids_with_data = np.sum(has_data_mask)
                        total_ids = len(region_ids_in_file)

                        logger.debug(f"Date {date_str}: {ids_with_data}/{total_ids} IDs have data")

                        if ids_with_data == total_ids:
                            # All IDs have data
                            present_dates.append(date_str)
                        elif ids_with_data > 0:
                            # Some IDs have data, some don't
                            partial_dates.append(date_str)
                            missing_dates.append(date_str)
                        else:
                            # No IDs have data
                            missing_dates.append(date_str)
                    else:
                        # Variable not in this subset - assume present
                        logger.warning(f"{data_var} not found in date subset, assuming date {date_str} is present")
                        present_dates.append(date_str)

                except Exception as e:
                    logger.warning(f"Error checking date {date_str}: {e}")
                    missing_dates.append(date_str)
            else:
                missing_dates.append(date_str)

        # Determine status
        all_present = len(missing_dates) == 0 and len(partial_dates) == 0
        some_present = len(present_dates) > 0 or len(partial_dates) > 0
        none_present = len(present_dates) == 0 and len(partial_dates) == 0

        ds.close()

        result = {
            'all_dates_present': all_present,
            'some_dates_present': some_present,
            'none_dates_present': none_present,
            'present_dates': present_dates,
            'missing_dates': missing_dates,
            'partial_dates': partial_dates,
            'file_path': historical_file_path,
            'region_found': True,
            'region_has_ids': True,
            'region_id_count': len(region_ids),
            'region_ids_in_file': len(region_ids_in_file),
            'region_ids_missing': len(region_ids_missing),
            'no_data_expected': False,
            'data_var_used': data_var
        }

        # Log summary
        logger.info(f"\n{'=' * 60}")
        logger.info(f"REGION CHECK: {region}")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total IDs in region: {result['region_id_count']}")
        logger.info(f"IDs in file: {result['region_ids_in_file']}")
        logger.info(f"IDs missing: {result['region_ids_missing']}")
        logger.info(f"Data variable used: {data_var}")
        logger.info(f"All dates present: {result['all_dates_present']}")
        logger.info(f"Some dates present: {result['some_dates_present']}")
        logger.info(f"None dates present: {result['none_dates_present']}")
        logger.info(f"Present dates: {result['present_dates']}")
        logger.info(f"Partial dates: {result['partial_dates']}")
        logger.info(f"Missing dates: {result['missing_dates']}")

        return result

    except Exception as e:
        logger.error(f"Error checking region dates: {e}")
        import traceback
        traceback.print_exc()
        return {
            'all_dates_present': False,
            'some_dates_present': False,
            'none_dates_present': True,
            'error': str(e),
            'missing_dates': dates_to_check,
            'present_dates': [],
            'partial_dates': [],
            'file_path': historical_file_path,
            'region_found': False,
            'region_has_ids': False,
            'no_data_expected': False
        }

def debug_historical_dates(historical_file_path: str) -> None:
    """
    Debug function to print dates from a historical NetCDF file.
    Call this before any processing to see what dates are available.
    """
    import xarray as xr
    import pandas as pd

    print("\n" + "=" * 80)
    print("DEBUG: HISTORICAL FILE DATES")
    print("=" * 80)
    print(f"File: {historical_file_path}")

    try:
        ds = xr.open_dataset(historical_file_path)

        if 'date' in ds.coords:
            hist_dates = pd.to_datetime(ds.date.values)
            hist_date_strings = [d.strftime("%Y-%m-%d") for d in hist_dates]

            print(f"Total dates: {len(hist_dates)}")
            print(f"Date range: {hist_date_strings[0]} to {hist_date_strings[-1]}")
            print(f"First 10 dates: {hist_date_strings[:10]}")
            print(f"Last 10 dates: {hist_date_strings[-10:]}")

            # Get unique months
            months = sorted(set([d.strftime("%Y-%m") for d in hist_dates]))
            print(f"Total unique months: {len(months)}")
            print(f"First 20 months: {months[:20]}")
            print(f"Last 20 months: {months[-20:]}")

            # Check for specific years
            for year in [2024, 2025, 2026]:
                year_months = [m for m in months if m.startswith(str(year))]
                print(f"  {year}: {len(year_months)} months - {year_months}")

            # Check date format
            print(f"\nDate format example: {hist_date_strings[0]}")
            print(f"Date type: {type(hist_dates[0])}")

        else:
            print("⚠️ No 'date' coordinate found in dataset")
            print(f"Available coordinates: {list(ds.coords.keys())}")
            print(f"Available dimensions: {list(ds.dims.keys())}")
            print(f"Available data variables: {list(ds.data_vars.keys())}")

        ds.close()

    except Exception as e:
        print(f"❌ Error reading historical file: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 80)


def process_region_date_new_fast_NRT(
        region: str,
        analysis_date: str,
        env_path: str = None,
        id_chunk_size: int = 500,  # Reduced from 2000 for memory
        n_jobs: int = 8,  # Reduced from 12 for memory
        save_interval: int = 1,  # Save every chunk
) -> Dict[str, Any]:
    """
    Process a single date for a region using batch processing for speed.
    Uses NRTBreakpoint.calculate_break with a LIST of IDs for internal parallelization.
    Memory-optimized version: saves incrementally and clears memory after each chunk.
    """
    import time
    import os
    import geopandas as gpd

    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    if n_jobs is None:
        n_jobs = min(8, os.cpu_count() or 1)  # Cap at 8 to prevent memory issues
    logger.info(f"Using {n_jobs} parallel jobs for processing (internal to NRTBreakpoint)")

    logger.info(f"\n{'=' * 80}")
    logger.info(f"PROCESSING {region} FOR DATE: {analysis_date} (FAST MODE - MEMORY OPTIMIZED)")
    logger.info(f"{'=' * 80}")

    try:
        # 1. Setup paths
        dynamic_world_data_dir = os.environ.get('dynamic_world_data')
        if not dynamic_world_data_dir:
            logger.error("❌ dynamic_world_data not set in environment")
            return {'success': False, 'error': 'dynamic_world_data not set in environment'}

        merge_dir = Path(dynamic_world_data_dir) / 'merge'
        nc_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
        historical_file = max(nc_files, key=os.path.getmtime)
        new_data_file = merge_dir / f"dw_{region}_{analysis_date}.nc"
        vector_lake_file = os.environ.get('vector_lake_file')

        # 2. Get region boundaries
        from utils.region_boundaries import get_region_boundaries
        region_boundaries = get_region_boundaries()

        if region not in region_boundaries:
            logger.error(f"❌ Region {region} not found in boundaries!")
            return {'success': False, 'error': f'Region {region} not found in boundaries'}

        bounds = region_boundaries[region]
        logger.info(f"📍 Region boundaries for {region}:")
        logger.info(f"   X_MIN_START: {bounds['X_MIN_START']}")
        logger.info(f"   X_MIN_END: {bounds['X_MIN_END']}")
        logger.info(f"   Y_MIN_START: {bounds['Y_MIN_START']}")
        logger.info(f"   Y_MIN_END: {bounds['Y_MIN_END']}")

        # 3. Load vector file and filter by region boundaries
        logger.info(f"Loading vector file and filtering for region {region}...")

        if not vector_lake_file or not Path(vector_lake_file).exists():
            logger.error(f"❌ Vector file not found: {vector_lake_file}")
            return {'success': False, 'error': 'Vector file not found'}

        gdf = gpd.read_parquet(vector_lake_file)

        # Handle Polygon geometries - convert to centroids for filtering
        if gdf.geometry.geom_type.iloc[0] in ['Polygon', 'MultiPolygon']:
            logger.info("Converting Polygon geometries to centroids for spatial filtering")
            gdf['centroid'] = gdf.geometry.centroid
            gdf = gdf.set_geometry('centroid')

        # Filter by region boundaries
        region_ids = gdf[
            (gdf.geometry.x >= bounds['X_MIN_START']) &
            (gdf.geometry.x <= bounds['X_MIN_END']) &
            (gdf.geometry.y >= bounds['Y_MIN_START']) &
            (gdf.geometry.y <= bounds['Y_MIN_END'])
            ]['id_geohash'].values

        region_ids_list = list(region_ids)
        logger.info(f"📍 {region} has {len(region_ids_list):,} IDs in vector file")

        if len(region_ids_list) == 0:
            logger.error(f"❌ No IDs found for region {region} in vector file!")
            return {'success': False, 'error': f'No IDs found for region {region}'}

        # 4. Load historical data - ONLY for the region IDs
        logger.info(f"Loading historical data for training (only {len(region_ids_list):,} IDs)...")

        if not historical_file.exists():
            logger.error(f"❌ Historical file not found: {historical_file}")
            return {'success': False, 'error': 'Historical file not found'}

        ds_historical_full = xr.open_dataset(str(historical_file))

        # Select only the region IDs (only those that exist in historical data)
        try:
            hist_ids = set(ds_historical_full.id_geohash.values)
            valid_region_ids = [id_val for id_val in region_ids_list if id_val in hist_ids]

            if len(valid_region_ids) != len(region_ids_list):
                logger.warning(
                    f"Filtered {len(region_ids_list) - len(valid_region_ids)} IDs not found in historical data")
                region_ids_list = valid_region_ids

            if len(region_ids_list) == 0:
                logger.error("❌ No valid region IDs found in historical data!")
                ds_historical_full.close()
                return {'success': False, 'error': 'No valid region IDs found in historical data'}

            ds_historical = ds_historical_full.sel(id_geohash=region_ids_list)
            logger.info(f"Loaded historical data for {len(ds_historical.id_geohash)} IDs")
        except Exception as e:
            logger.error(f"Error selecting region IDs from historical file: {e}")
            ds_historical_full.close()
            return {'success': False, 'error': f'Error loading historical data: {e}'}

        ds_historical_full.close()
        del ds_historical_full
        gc.collect()

        analysis_timestamp = pd.Timestamp(f"{analysis_date}-01")
        ds_historical_train = ds_historical.where(ds_historical.date < analysis_timestamp, drop=True)

        if len(ds_historical_train.date) == 0:
            logger.warning(f"No training data before {analysis_date}, using all historical data")
            ds_historical_train = ds_historical

        logger.info(f"Training data has {len(ds_historical_train.date)} dates")
        logger.info(f"Training data has {len(ds_historical_train.id_geohash)} IDs")

        # 5. Determine where the analysis data comes from
        analysis_source = None
        ds_analysis = None

        if new_data_file.exists():
            try:
                ds_analysis = xr.open_dataset(str(new_data_file))
                if 'date' in ds_analysis.coords:
                    dates_in_file = pd.to_datetime(ds_analysis.date.values)
                    date_strings = [d.strftime("%Y-%m") for d in dates_in_file]
                    if analysis_date in date_strings:
                        analysis_source = 'downloaded'
                        logger.info(f"📊 Using DOWNLOADED data for {analysis_date} from: {new_data_file}")
                        logger.info(f"   IDs in downloaded data: {len(ds_analysis.id_geohash)}")
                    else:
                        ds_analysis.close()
                        ds_analysis = None
            except Exception as e:
                logger.warning(f"Error reading new data file: {e}")
                if ds_analysis:
                    ds_analysis.close()
                    ds_analysis = None

        if ds_analysis is None and historical_file.exists():
            try:
                ds_historical_check = xr.open_dataset(str(historical_file))
                if 'date' in ds_historical_check.coords:
                    dates_in_file = pd.to_datetime(ds_historical_check.date.values)
                    date_strings = [d.strftime("%Y-%m") for d in dates_in_file]
                    if analysis_date in date_strings:
                        ds_analysis = ds_historical_check.sel(id_geohash=region_ids_list, date=analysis_timestamp)
                        analysis_source = 'historical'
                        logger.info(f"📊 Using HISTORICAL data for {analysis_date} from: {historical_file}")
                        logger.info(f"   IDs in historical analysis data: {len(ds_analysis.id_geohash)}")
                    else:
                        logger.warning(f"Date {analysis_date} not found in historical file")
                ds_historical_check.close()
            except Exception as e:
                logger.error(f"Error reading historical file: {e}")

        if ds_analysis is None:
            logger.error(f"❌ No data found for {region} {analysis_date} in either downloaded or historical files")
            ds_historical.close()
            return {'success': False, 'error': f'No data found for {region} {analysis_date}'}

        # 6. Filter analysis data to only IDs that exist in region_ids_list
        analysis_ids = set(ds_analysis.id_geohash.values) if 'id_geohash' in ds_analysis.dims else set()
        region_ids_set = set(region_ids_list)
        common_ids = analysis_ids.intersection(region_ids_set)

        if len(common_ids) < len(region_ids_list):
            logger.info(f"Filtered {len(region_ids_list) - len(common_ids)} IDs not found in analysis data")
            region_ids_list = list(common_ids)
            logger.info(f"Updated region IDs: {len(region_ids_list):,}")

        if len(region_ids_list) == 0:
            logger.error(f"❌ No overlapping IDs between region and analysis data!")
            ds_historical.close()
            ds_analysis.close()
            return {'success': False, 'error': 'No overlapping IDs found'}

        # Now filter ds_analysis to only region IDs
        if 'id_geohash' in ds_analysis.dims:
            ds_analysis = ds_analysis.sel(id_geohash=region_ids_list)

        # 7. Get matching IDs (only region IDs that exist in both datasets)
        train_ids = set(ds_historical_train.id_geohash.values) if 'id_geohash' in ds_historical_train.dims else set()
        analysis_ids = set(ds_analysis.id_geohash.values) if 'id_geohash' in ds_analysis.dims else set()

        matching_ids = train_ids.intersection(analysis_ids)
        logger.info(f"📊 ID Summary:")
        logger.info(f"   IDs in training data: {len(train_ids):,}")
        logger.info(f"   IDs in analysis data: {len(analysis_ids):,}")
        logger.info(f"   Matching IDs: {len(matching_ids):,}")

        if len(matching_ids) == 0:
            logger.error(f"No matching IDs found between training and analysis data!")
            ds_historical.close()
            ds_analysis.close()
            return {'success': False, 'error': 'No matching IDs found'}

        # 8. Filter datasets to matching IDs
        matching_ids_list = list(matching_ids)
        ds_historical_train = ds_historical_train.sel(id_geohash=matching_ids_list)
        ds_analysis = ds_analysis.sel(id_geohash=matching_ids_list)

        logger.info(f"Filtered training data to {len(ds_historical_train.id_geohash)} IDs")
        logger.info(f"Filtered analysis data to {len(ds_analysis.id_geohash)} IDs")

        # 9. Setup output directories
        output_dir = os.environ.get('output_dir')
        if not output_dir:
            logger.error("❌ output_dir not set in environment")
            return {'success': False, 'error': 'output_dir not set in environment'}

        output_dir = Path(output_dir) / region
        zarr_output_dir = output_dir / 'breakpoint_zarr'
        zarr_output_dir.mkdir(exist_ok=True, parents=True)
        zarr_path = zarr_output_dir / f'breakpoints_{analysis_date}.zarr'

        current_breakpoint_dir = output_dir / f'breakpoint_{analysis_date}'
        current_breakpoint_dir.mkdir(exist_ok=True, parents=True)
        intermediate_file = current_breakpoint_dir / f'intermediate_results_{analysis_date}.parquet'

        # NEW: File for accumulating results incrementally
        incremental_file = current_breakpoint_dir / f'incremental_results_{analysis_date}.parquet'

        # 10. Process in chunks - MEMORY OPTIMIZED
        bp = NRTBreakpoint()
        total_processed = 0
        total_breakpoints = 0
        total_ids = len(matching_ids_list)
        analysis_date_str = analysis_timestamp.strftime("%Y-%m-%d")

        logger.info(f"Using analysis date: {analysis_date_str}")
        logger.info(f"Starting processing of {total_ids:,} IDs in chunks of {id_chunk_size}")
        logger.info(f"NOTE: Each chunk will be passed to NRTBreakpoint.calculate_break as a LIST")
        logger.info(f"       This enables internal parallelization with {n_jobs} workers")
        logger.info(f"Results will be saved incrementally after EACH chunk (memory-optimized)")

        # Check for existing incremental file (resume capability)
        if incremental_file.exists():
            try:
                saved_results = pd.read_parquet(incremental_file)
                saved_ids = set(saved_results['id_geohash'].values)
                matching_ids_list = [id_val for id_val in matching_ids_list if id_val not in saved_ids]
                total_breakpoints = len(saved_results)
                logger.info(f"🔄 Resuming from incremental file: {len(saved_ids)} IDs already processed")
                logger.info(f"   Remaining IDs: {len(matching_ids_list)}")
                logger.info(f"   Existing breakpoints: {total_breakpoints:,}")
            except Exception as e:
                logger.warning(f"Error reading incremental file, starting fresh: {e}")
                if incremental_file.exists():
                    incremental_file.unlink()

        start_time = time.time()
        processed_since_last_save = 0

        all_ids = list(matching_ids_list)
        total_chunks = (len(all_ids) + id_chunk_size - 1) // id_chunk_size if all_ids else 0

        if total_chunks == 0:
            logger.warning("No chunks to process")
            # Check if we have existing data to create Zarr
            if incremental_file.exists():
                logger.info("Using existing incremental data to create Zarr")
                return create_final_zarr_from_incremental(
                    incremental_file, zarr_path, region, analysis_date, analysis_source, total_ids
                )
            return {'success': True, 'total_ids': total_ids, 'processed': 0, 'breakpoints_found': 0,
                    'zarr_path': str(zarr_path)}

        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * id_chunk_size
            end_idx = min(start_idx + id_chunk_size, len(all_ids))
            chunk_ids = all_ids[start_idx:end_idx]
            chunk_start_time = time.time()
            progress_pct = (float(total_processed) / float(total_ids))
            logger.info(f"Chunk {chunk_idx + 1}/{total_chunks}: {len(chunk_ids)} IDs ({progress_pct:.1f}%)")

            try:
                # Get data for this chunk
                ds_historical_chunk = ds_historical_train.sel(id_geohash=chunk_ids)
                ds_analysis_chunk = ds_analysis.sel(id_geohash=chunk_ids)

                ds_combined = xr.concat([ds_historical_chunk, ds_analysis_chunk], dim='date')
                ds_combined = ds_combined.sortby('date')
                dwds = DWDataset(ds_combined)

                # Pass the ENTIRE LIST of IDs to calculate_break
                breaks_df = bp.calculate_break(
                    dataset=dwds,
                    analysis_date=analysis_date_str,
                    object_id=chunk_ids,  # <-- PASS THE LIST!
                    keep_nans=False
                )

                if breaks_df is not None and not breaks_df.empty:
                    # Ensure id_geohash is a column
                    if 'id_geohash' not in breaks_df.columns:
                        breaks_df = breaks_df.reset_index()

                    # SAVE INCREMENTALLY IMMEDIATELY (don't accumulate in memory)
                    if incremental_file.exists():
                        # Append to existing file
                        existing = pd.read_parquet(incremental_file)
                        combined = pd.concat([existing, breaks_df], ignore_index=True)
                        combined.to_parquet(incremental_file)
                    else:
                        # First save
                        breaks_df.to_parquet(incremental_file)

                    total_breakpoints += len(breaks_df)

                    # Clear breaks_df from memory
                    del breaks_df

                    logger.info(
                        f"  ✅ Chunk {chunk_idx + 1} complete: {len(breaks_df)} breakpoints found (saved incrementally)")
                else:
                    logger.info(f"  ✅ Chunk {chunk_idx + 1} complete: 0 breakpoints found")

                total_processed += len(chunk_ids)

                # Log chunk timing
                chunk_time = time.time() - chunk_start_time
                ids_per_second = len(chunk_ids) / chunk_time if chunk_time > 0 else 0
                logger.info(f"  ⏱️ Chunk {chunk_idx + 1} took {chunk_time:.1f}s ({ids_per_second:.1f} IDs/sec)")

                # Memory cleanup after each chunk
                del ds_historical_chunk, ds_analysis_chunk, ds_combined, dwds
                gc.collect()

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                # Continue to next chunk instead of failing completely
                continue

        # 11. Create final Zarr file from incremental data
        final_result = create_final_zarr_from_incremental(
            incremental_file, zarr_path, region, analysis_date, analysis_source, total_ids
        )

        # Clean up
        ds_historical.close()
        ds_analysis.close()
        gc.collect()

        total_time = time.time() - start_time
        minutes, seconds = divmod(total_time, 60)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"📊 FINAL SUMMARY for {region} {analysis_date}")
        logger.info(f"{'=' * 60}")
        logger.info(f"   Total IDs processed: {total_processed:,}")
        logger.info(f"   Total breakpoints found: {total_breakpoints:,}")
        logger.info(f"   Total chunks: {total_chunks}")
        logger.info(f"   Total time: {int(minutes)}m {int(seconds)}s")
        if total_processed > 0:
            logger.info(f"   Average time per ID: {total_time / total_processed:.2f}s")
        logger.info(f"{'=' * 60}")

        return final_result

    except Exception as e:
        logger.error(f"❌ Unexpected error in process_region_date_new_fast_NRT: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'region': region,
            'analysis_date': analysis_date
        }

def process_region_date_new_fast_historical(
        region: str,
        analysis_date: str,
        env_path: str = None,
        id_chunk_size: int = 500,  # Reduced from 2000 for memory
        n_jobs: int = 8,  # Reduced from 12 for memory
        save_interval: int = 1,  # Save every chunk
) -> Dict[str, Any]:
    """
    Process a single date for a region using batch processing for speed.
    Uses NRTBreakpoint.calculate_break with a LIST of IDs for internal parallelization.
    Memory-optimized version: saves incrementally and clears memory after each chunk.
    """
    import time
    import os
    import geopandas as gpd

    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    if n_jobs is None:
        n_jobs = min(8, os.cpu_count() or 1)  # Cap at 8 to prevent memory issues
    logger.info(f"Using BeastBreakpoint (RBEAST) for historical breakpoint detection")

    logger.info(f"\n{'=' * 80}")
    logger.info(f"PROCESSING {region} FOR DATE: {analysis_date} (FAST MODE - MEMORY OPTIMIZED)")
    logger.info(f"{'=' * 80}")

    try:
        # 1. Setup paths
        dynamic_world_data_dir = os.environ.get('dynamic_world_data')
        if not dynamic_world_data_dir:
            logger.error("❌ dynamic_world_data not set in environment")
            return {'success': False, 'error': 'dynamic_world_data not set in environment'}

        merge_dir = Path(dynamic_world_data_dir) / 'merge'
        nc_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
        historical_file = max(nc_files, key=os.path.getmtime)
        new_data_file = merge_dir / f"dw_{region}_{analysis_date}.nc"
        vector_lake_file = os.environ.get('vector_lake_file')

        # 2. Get region boundaries
        from utils.region_boundaries import get_region_boundaries
        region_boundaries = get_region_boundaries()

        if region not in region_boundaries:
            logger.error(f"❌ Region {region} not found in boundaries!")
            return {'success': False, 'error': f'Region {region} not found in boundaries'}

        bounds = region_boundaries[region]
        logger.info(f"📍 Region boundaries for {region}:")
        logger.info(f"   X_MIN_START: {bounds['X_MIN_START']}")
        logger.info(f"   X_MIN_END: {bounds['X_MIN_END']}")
        logger.info(f"   Y_MIN_START: {bounds['Y_MIN_START']}")
        logger.info(f"   Y_MIN_END: {bounds['Y_MIN_END']}")

        # 3. Load vector file and filter by region boundaries
        logger.info(f"Loading vector file and filtering for region {region}...")

        if not vector_lake_file or not Path(vector_lake_file).exists():
            logger.error(f"❌ Vector file not found: {vector_lake_file}")
            return {'success': False, 'error': 'Vector file not found'}

        gdf = gpd.read_parquet(vector_lake_file)

        # Handle Polygon geometries - convert to centroids for filtering
        if gdf.geometry.geom_type.iloc[0] in ['Polygon', 'MultiPolygon']:
            logger.info("Converting Polygon geometries to centroids for spatial filtering")
            gdf['centroid'] = gdf.geometry.centroid
            gdf = gdf.set_geometry('centroid')

        # Filter by region boundaries
        region_ids = gdf[
            (gdf.geometry.x >= bounds['X_MIN_START']) &
            (gdf.geometry.x <= bounds['X_MIN_END']) &
            (gdf.geometry.y >= bounds['Y_MIN_START']) &
            (gdf.geometry.y <= bounds['Y_MIN_END'])
            ]['id_geohash'].values

        region_ids_list = list(region_ids)
        logger.info(f"📍 {region} has {len(region_ids_list):,} IDs in vector file")

        if len(region_ids_list) == 0:
            logger.error(f"❌ No IDs found for region {region} in vector file!")
            return {'success': False, 'error': f'No IDs found for region {region}'}

        # 4. Load historical data - ONLY for the region IDs
        logger.info(f"Loading historical data for training (only {len(region_ids_list):,} IDs)...")

        if not historical_file.exists():
            logger.error(f"❌ Historical file not found: {historical_file}")
            return {'success': False, 'error': 'Historical file not found'}

        ds_historical_full = xr.open_dataset(str(historical_file))

        # Select only the region IDs (only those that exist in historical data)
        try:
            hist_ids = set(ds_historical_full.id_geohash.values)
            valid_region_ids = [id_val for id_val in region_ids_list if id_val in hist_ids]

            if len(valid_region_ids) != len(region_ids_list):
                logger.warning(
                    f"Filtered {len(region_ids_list) - len(valid_region_ids)} IDs not found in historical data")
                region_ids_list = valid_region_ids

            if len(region_ids_list) == 0:
                logger.error("❌ No valid region IDs found in historical data!")
                ds_historical_full.close()
                return {'success': False, 'error': 'No valid region IDs found in historical data'}

            ds_historical = ds_historical_full.sel(id_geohash=region_ids_list)
            logger.info(f"Loaded historical data for {len(ds_historical.id_geohash)} IDs")
        except Exception as e:
            logger.error(f"Error selecting region IDs from historical file: {e}")
            ds_historical_full.close()
            return {'success': False, 'error': f'Error loading historical data: {e}'}

        ds_historical_full.close()
        del ds_historical_full
        gc.collect()

        analysis_timestamp = pd.Timestamp(f"{analysis_date}-01")
        ds_historical_train = ds_historical.where(ds_historical.date < analysis_timestamp, drop=True)

        if len(ds_historical_train.date) == 0:
            logger.warning(f"No training data before {analysis_date}, using all historical data")
            ds_historical_train = ds_historical

        logger.info(f"Training data has {len(ds_historical_train.date)} dates")
        logger.info(f"Training data has {len(ds_historical_train.id_geohash)} IDs")

        # 5. Determine where the analysis data comes from
        analysis_source = None
        ds_analysis = None

        if ds_analysis is None and historical_file.exists():
            try:
                ds_historical_check = xr.open_dataset(str(historical_file))
                if 'date' in ds_historical_check.coords:
                    dates_in_file = pd.to_datetime(ds_historical_check.date.values)
                    date_strings = [d.strftime("%Y-%m") for d in dates_in_file]
                    if analysis_date in date_strings:
                        ds_analysis = ds_historical_check.sel(id_geohash=region_ids_list, date=analysis_timestamp)
                        analysis_source = 'historical'
                        logger.info(f"📊 Using HISTORICAL data for {analysis_date} from: {historical_file}")
                        logger.info(f"   IDs in historical analysis data: {len(ds_analysis.id_geohash)}")
                    else:
                        logger.warning(f"Date {analysis_date} not found in historical file")
                ds_historical_check.close()
            except Exception as e:
                logger.error(f"Error reading historical file: {e}")

        if ds_analysis is None:
            logger.error(f"❌ No data found for {region} {analysis_date} in either downloaded or historical files")
            ds_historical.close()
            return {'success': False, 'error': f'No data found for {region} {analysis_date}'}

        # 6. Filter analysis data to only IDs that exist in region_ids_list
        analysis_ids = set(ds_analysis.id_geohash.values) if 'id_geohash' in ds_analysis.dims else set()
        region_ids_set = set(region_ids_list)
        common_ids = analysis_ids.intersection(region_ids_set)

        if len(common_ids) < len(region_ids_list):
            logger.info(f"Filtered {len(region_ids_list) - len(common_ids)} IDs not found in analysis data")
            region_ids_list = list(common_ids)
            logger.info(f"Updated region IDs: {len(region_ids_list):,}")

        if len(region_ids_list) == 0:
            logger.error(f"❌ No overlapping IDs between region and analysis data!")
            ds_historical.close()
            ds_analysis.close()
            return {'success': False, 'error': 'No overlapping IDs found'}

        # Now filter ds_analysis to only region IDs
        if 'id_geohash' in ds_analysis.dims:
            ds_analysis = ds_analysis.sel(id_geohash=region_ids_list)

        # 7. Get matching IDs (only region IDs that exist in both datasets)
        train_ids = set(ds_historical_train.id_geohash.values) if 'id_geohash' in ds_historical_train.dims else set()
        analysis_ids = set(ds_analysis.id_geohash.values) if 'id_geohash' in ds_analysis.dims else set()

        matching_ids = train_ids.intersection(analysis_ids)
        logger.info(f"📊 ID Summary:")
        logger.info(f"   IDs in training data: {len(train_ids):,}")
        logger.info(f"   IDs in analysis data: {len(analysis_ids):,}")
        logger.info(f"   Matching IDs: {len(matching_ids):,}")

        if len(matching_ids) == 0:
            logger.error(f"No matching IDs found between training and analysis data!")
            ds_historical.close()
            ds_analysis.close()
            return {'success': False, 'error': 'No matching IDs found'}

        # 8. Filter datasets to matching IDs
        matching_ids_list = list(matching_ids)
        ds_historical_train = ds_historical_train.sel(id_geohash=matching_ids_list)
        ds_analysis = ds_analysis.sel(id_geohash=matching_ids_list)

        logger.info(f"Filtered training data to {len(ds_historical_train.id_geohash)} IDs")
        logger.info(f"Filtered analysis data to {len(ds_analysis.id_geohash)} IDs")

        # 9. Setup output directories
        output_dir = os.environ.get('output_dir')
        if not output_dir:
            logger.error("❌ output_dir not set in environment")
            return {'success': False, 'error': 'output_dir not set in environment'}

        output_dir = Path(output_dir) / region
        zarr_output_dir = output_dir / 'breakpoint_zarr'
        zarr_output_dir.mkdir(exist_ok=True, parents=True)
        zarr_path = zarr_output_dir / f'breakpoints_{analysis_date}.zarr'

        current_breakpoint_dir = output_dir / f'breakpoint_{analysis_date}'
        current_breakpoint_dir.mkdir(exist_ok=True, parents=True)
        intermediate_file = current_breakpoint_dir / f'intermediate_results_{analysis_date}.parquet'

        # NEW: File for accumulating results incrementally
        incremental_file = current_breakpoint_dir / f'incremental_results_{analysis_date}.parquet'

        # 10. Process in chunks - MEMORY OPTIMIZED
        bp = BeastBreakpoint()
        total_processed = 0
        total_breakpoints = 0
        total_ids = len(matching_ids_list)
        analysis_date_str = analysis_timestamp.strftime("%Y-%m-%d")

        logger.info(f"Using analysis date: {analysis_date_str}")
        logger.info(f"Starting processing of {total_ids:,} IDs in chunks of {id_chunk_size}")
        logger.info(f"NOTE: Each chunk is passed to BeastBreakpoint.calculate_breaks_batch")
        logger.info(f"       BEAST (RBEAST) detects breakpoints across the full historical series per lake")
        logger.info(f"Results will be saved incrementally after EACH chunk (memory-optimized)")

        # Check for existing incremental file (resume capability)
        if incremental_file.exists():
            try:
                saved_results = pd.read_parquet(incremental_file)
                saved_ids = set(saved_results['id_geohash'].values)
                matching_ids_list = [id_val for id_val in matching_ids_list if id_val not in saved_ids]
                total_breakpoints = len(saved_results)
                logger.info(f"🔄 Resuming from incremental file: {len(saved_ids)} IDs already processed")
                logger.info(f"   Remaining IDs: {len(matching_ids_list)}")
                logger.info(f"   Existing breakpoints: {total_breakpoints:,}")
            except Exception as e:
                logger.warning(f"Error reading incremental file, starting fresh: {e}")
                if incremental_file.exists():
                    incremental_file.unlink()

        start_time = time.time()
        processed_since_last_save = 0

        all_ids = list(matching_ids_list)
        total_chunks = (len(all_ids) + id_chunk_size - 1) // id_chunk_size if all_ids else 0

        if total_chunks == 0:
            logger.warning("No chunks to process")
            # Check if we have existing data to create Zarr
            if incremental_file.exists():
                logger.info("Using existing incremental data to create Zarr")
                return create_final_zarr_from_incremental(
                    incremental_file, zarr_path, region, analysis_date, analysis_source, total_ids
                )
            return {'success': True, 'total_ids': total_ids, 'processed': 0, 'breakpoints_found': 0,
                    'zarr_path': str(zarr_path)}

        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * id_chunk_size
            end_idx = min(start_idx + id_chunk_size, len(all_ids))
            chunk_ids = all_ids[start_idx:end_idx]
            chunk_start_time = time.time()
            progress_pct = (float(total_processed) / float(total_ids))
            logger.info(f"Chunk {chunk_idx + 1}/{total_chunks}: {len(chunk_ids)} IDs ({progress_pct:.1f}%)")

            try:
                # Get data for this chunk
                ds_historical_chunk = ds_historical_train.sel(id_geohash=chunk_ids)
                ds_analysis_chunk = ds_analysis.sel(id_geohash=chunk_ids)

                ds_combined = xr.concat([ds_historical_chunk, ds_analysis_chunk], dim='date')
                ds_combined = ds_combined.sortby('date')
                dwds = DWDataset(ds_combined)

                # dwds already contains only chunk_ids; run BEAST over every lake in the chunk
                breaks_df = bp.calculate_breaks_batch(dataset=dwds)

                if breaks_df is not None and not breaks_df.empty:
                    # Ensure id_geohash is a column
                    if 'id_geohash' not in breaks_df.columns:
                        breaks_df = breaks_df.reset_index()

                    # SAVE INCREMENTALLY IMMEDIATELY (don't accumulate in memory)
                    if incremental_file.exists():
                        # Append to existing file
                        existing = pd.read_parquet(incremental_file)
                        combined = pd.concat([existing, breaks_df], ignore_index=True)
                        combined.to_parquet(incremental_file)
                    else:
                        # First save
                        breaks_df.to_parquet(incremental_file)

                    total_breakpoints += len(breaks_df)

                    # Clear breaks_df from memory
                    del breaks_df

                    logger.info(
                        f"  ✅ Chunk {chunk_idx + 1} complete: {len(breaks_df)} breakpoints found (saved incrementally)")
                else:
                    logger.info(f"  ✅ Chunk {chunk_idx + 1} complete: 0 breakpoints found")

                total_processed += len(chunk_ids)

                # Log chunk timing
                chunk_time = time.time() - chunk_start_time
                ids_per_second = len(chunk_ids) / chunk_time if chunk_time > 0 else 0
                logger.info(f"  ⏱️ Chunk {chunk_idx + 1} took {chunk_time:.1f}s ({ids_per_second:.1f} IDs/sec)")

                # Memory cleanup after each chunk
                del ds_historical_chunk, ds_analysis_chunk, ds_combined, dwds
                gc.collect()

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                # Continue to next chunk instead of failing completely
                continue

        # 11. Create final Zarr file from incremental data
        final_result = create_final_zarr_from_incremental(
            incremental_file, zarr_path, region, analysis_date, analysis_source, total_ids
        )

        # Clean up
        ds_historical.close()
        ds_analysis.close()
        gc.collect()

        total_time = time.time() - start_time
        minutes, seconds = divmod(total_time, 60)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"📊 FINAL SUMMARY for {region} {analysis_date}")
        logger.info(f"{'=' * 60}")
        logger.info(f"   Total IDs processed: {total_processed:,}")
        logger.info(f"   Total breakpoints found: {total_breakpoints:,}")
        logger.info(f"   Total chunks: {total_chunks}")
        logger.info(f"   Total time: {int(minutes)}m {int(seconds)}s")
        if total_processed > 0:
            logger.info(f"   Average time per ID: {total_time / total_processed:.2f}s")
        logger.info(f"{'=' * 60}")

        return final_result

    except Exception as e:
        logger.error(f"❌ Unexpected error in process_region_date_new_fast_NRT: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'region': region,
            'analysis_date': analysis_date
        }


def create_final_zarr_from_incremental(
        incremental_file: Path,
        zarr_path: Path,
        region: str,
        analysis_date: str,
        analysis_source: str,
        total_ids: int
) -> Dict[str, Any]:
    """
    Create the final Zarr file from incremental results.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"📦 CREATING FINAL ZARR FROM INCREMENTAL DATA")
    logger.info(f"{'=' * 60}")

    if not incremental_file.exists():
        logger.warning(f"No incremental file found at {incremental_file}")
        # Create empty Zarr
        empty_result = pd.DataFrame(columns=[
            'id_geohash', 'date', 'water_observed', 'water_predicted', 'water_residual',
            'water_predicted_lower_90', 'water_predicted_upper_90',
            'water_historical_mean', 'water_historical_median',
            'water_historical_std', 'water_historical_min',
            'water_historical_max', 'drainage_confidence'
        ])
        empty_ds = empty_result.set_index('id_geohash').to_xarray()
        empty_ds.attrs.update({
            'region': region,
            'analysis_date': analysis_date,
            'created_at': datetime.datetime.now().isoformat(),
            'complete': True,
            'empty': True,
            'analysis_source': analysis_source,
            'total_ids': total_ids
        })
        empty_ds.to_zarr(zarr_path, mode='w')
        empty_ds.close()
        return {
            'success': True,
            'total_ids': total_ids,
            'processed': 0,
            'breakpoints_found': 0,
            'zarr_path': str(zarr_path)
        }

    try:
        # Load all incremental results
        logger.info(f"Loading incremental results from {incremental_file}")
        breaks_merged = pd.read_parquet(incremental_file)

        if breaks_merged.empty:
            logger.warning("Incremental file is empty")
            # Create empty Zarr
            empty_result = pd.DataFrame(columns=[
                'id_geohash', 'date', 'water_observed', 'water_predicted', 'water_residual',
                'water_predicted_lower_90', 'water_predicted_upper_90',
                'water_historical_mean', 'water_historical_median',
                'water_historical_std', 'water_historical_min',
                'water_historical_max', 'drainage_confidence'
            ])
            empty_ds = empty_result.set_index('id_geohash').to_xarray()
            empty_ds.attrs.update({
                'region': region,
                'analysis_date': analysis_date,
                'created_at': datetime.datetime.now().isoformat(),
                'complete': True,
                'empty': True,
                'analysis_source': analysis_source,
                'total_ids': total_ids
            })
            empty_ds.to_zarr(zarr_path, mode='w')
            empty_ds.close()
            return {
                'success': True,
                'total_ids': total_ids,
                'processed': 0,
                'breakpoints_found': 0,
                'zarr_path': str(zarr_path)
            }

        # Save to Zarr
        logger.info(f"💾 Saving {len(breaks_merged):,} records to Zarr: {zarr_path}")

        # Set index and create dataset
        ds_breaks = breaks_merged.set_index('id_geohash').to_xarray()
        ds_breaks.attrs.update({
            'region': region,
            'analysis_date': analysis_date,
            'created_at': datetime.datetime.now().isoformat(),
            'complete': True,
            'analysis_source': analysis_source,
            'total_ids': total_ids,
            'breakpoints_found': len(breaks_merged)
        })

        ds_breaks.to_zarr(zarr_path, mode='w')
        logger.info(f"   ✅ Zarr saved successfully")

        # Optional: Save Parquet backup
        current_breakpoint_dir = incremental_file.parent
        path_to_joined_file = current_breakpoint_dir / f'drain_{analysis_date}.parquet'
        breaks_merged.to_parquet(path_to_joined_file)
        logger.info(f"   ✅ Parquet backup saved to {path_to_joined_file}")

        # Clean up incremental file (optional - keep for debugging)
        # incremental_file.unlink()
        # logger.info(f"   🧹 Incremental file cleaned up")

        if zarr_path.exists():
            zarr_size_gb = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file()) / (1024 ** 3)
            logger.info(f"   📦 Zarr file size: {zarr_size_gb:.2f} GB")

        ds_breaks.close()

        return {
            'success': True,
            'region': region,
            'analysis_date': analysis_date,
            'analysis_source': analysis_source,
            'total_ids': total_ids,
            'processed': len(breaks_merged),
            'breakpoints_found': len(breaks_merged),
            'zarr_path': str(zarr_path)
        }

    except Exception as e:
        logger.error(f"Error creating Zarr from incremental data: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'region': region,
            'analysis_date': analysis_date
        }


def verify_downloads_complete(
        region: str = "TEST",
        analysis_dates: List[str] = None,
        run_start_label: str = None,
        env_path: str = None,
        auto_discover_dates: bool = False,
        strict_mode: bool = True
):
    """
    Verify that all downloads for a region are complete for specified dates.
    Handles first-time runs gracefully.
    """
    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    REGION_NAME = region
    dynamic_world_download_dir = Path(os.environ['dynamic_world_downloads'])

    # ========== CREATE DIRECTORY IF IT DOESN'T EXIST ==========
    region_download_dir = dynamic_world_download_dir / REGION_NAME
    region_download_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists: {region_download_dir}")
    # ========== END DIRECTORY CREATION ==========

    # Normalize analysis_dates to string format "YYYY-MM"
    normalized_dates = []
    if analysis_dates is not None:
        for date in analysis_dates:
            if isinstance(date, pd.Timestamp):
                normalized_dates.append(date.strftime("%Y-%m"))
            elif isinstance(date, datetime.datetime):
                normalized_dates.append(date.strftime("%Y-%m"))
            elif isinstance(date, str):
                try:
                    if len(date) == 7 and date[4] == '-':
                        normalized_dates.append(date)
                    else:
                        dt = pd.to_datetime(date)
                        normalized_dates.append(dt.strftime("%Y-%m"))
                except:
                    logger.warning(f"Could not parse date: {date}")
            else:
                logger.warning(f"Unrecognized date type: {type(date)} for {date}")

    # Use normalized dates or discover
    if normalized_dates:
        analysis_dates = normalized_dates
    elif auto_discover_dates:
        download_pattern = str(dynamic_world_download_dir / REGION_NAME / 'download_*')
        download_dirs = glob.glob(download_pattern)

        discovered_dates = []
        for dir_path in download_dirs:
            dir_name = Path(dir_path).name
            if dir_name.startswith('download_'):
                date_str = dir_name.replace('download_', '')
                try:
                    datetime.datetime.strptime(date_str, '%Y-%m')
                    discovered_dates.append(date_str)
                except ValueError:
                    continue

        analysis_dates = sorted(discovered_dates)

        if not analysis_dates:
            return {
                'success': True,  # First run - nothing to verify yet
                'complete': False,
                'first_run': True,
                'error': None,
                'region': REGION_NAME,
                'reason': 'No dates found to verify (first run)',
                'need_download': True,  # Important: flag to trigger download
                'discovered_dates': discovered_dates,
                'date_results': {},
                'summary': {
                    'total_dates': 0,
                    'complete_count': 0,
                    'incomplete_count': 0,
                    'total_expected_downloads': 0,
                    'total_successful_downloads': 0,
                    'total_failed_downloads': 0,
                    'total_skipped_downloads': 0
                }
            }
    else:
        # No dates provided and auto_discover is False
        return {
            'success': False,
            'complete': False,
            'first_run': False,
            'error': 'No dates provided for verification',
            'region': REGION_NAME,
            'reason': 'No dates provided for verification',
            'need_download': False,
            'date_results': {},
            'summary': {
                'total_dates': 0,
                'complete_count': 0,
                'incomplete_count': 0,
                'total_expected_downloads': 0,
                'total_successful_downloads': 0,
                'total_failed_downloads': 0,
                'total_skipped_downloads': 0
            }
        }

    logger.info(f"Verifying downloads for region '{REGION_NAME}' for {len(analysis_dates)} date(s): {analysis_dates}")

    date_results = {}
    all_complete = True
    missing_dates = []
    directory_missing_dates = []
    no_manifest_dates = []
    first_run = True  # Track if any downloads exist

    for analysis_date in analysis_dates:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Verifying date: {analysis_date}")
        logger.info(f"{'=' * 60}")

        current_download_dir = dynamic_world_download_dir / REGION_NAME / f'download_{analysis_date}'

        date_result = {
            'analysis_date': analysis_date,
            'complete': False,
            'success': False,
            'expected_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'skipped_downloads': 0,
            'manifest_file': None,
            'completion_file': None,
            'merged_file': None,
            'details': {},
            'directory_exists': False,
            'manifest_exists': False
        }

        # Check if download directory exists
        if not current_download_dir.exists():
            logger.info(f"Download directory does not exist yet (first run): {current_download_dir}")
            date_result['directory_exists'] = False
            date_result['success'] = True  # Not a failure, just first run
            date_result['complete'] = False
            date_result['first_run'] = True
            date_result['reason'] = f"Download directory does not exist yet (first run)"
            date_results[analysis_date] = date_result
            all_complete = False
            missing_dates.append(analysis_date)
            directory_missing_dates.append(analysis_date)
            continue
        else:
            date_result['directory_exists'] = True
            first_run = False

        # Look for manifest files
        manifest_files = list(current_download_dir.glob(f'download_manifest_*.json'))
        if not manifest_files:
            logger.info(f"No manifest file found (first run): {analysis_date}")
            date_result['manifest_exists'] = False
            date_result['success'] = True  # Not a failure, just first run
            date_result['complete'] = False
            date_result['first_run'] = True
            date_result['reason'] = 'No manifest file found (first run)'
            date_results[analysis_date] = date_result
            all_complete = False
            missing_dates.append(analysis_date)
            no_manifest_dates.append(analysis_date)
            continue
        else:
            date_result['manifest_exists'] = True
            first_run = False

        # Get the most recent manifest
        manifest_file = max(manifest_files, key=lambda p: p.stat().st_mtime)
        with open(manifest_file, 'r') as f:
            manifest_data = json.load(f)

        date_result['manifest_file'] = str(manifest_file)
        date_result['expected_downloads'] = manifest_data.get('expected_downloads', 0)
        date_result['successful_downloads'] = manifest_data.get('successful_downloads', 0)
        date_result['failed_downloads'] = manifest_data.get('failed_downloads', 0)
        date_result['skipped_downloads'] = manifest_data.get('skipped_downloads', 0)
        date_result['manifest_data'] = manifest_data

        # Check if we have any expected downloads
        if date_result['expected_downloads'] == 0:
            logger.info(f"No expected downloads for {analysis_date} (no lakes in region)")
            date_result['reason'] = 'No expected downloads (no lakes in region)'
            date_result['success'] = True
            date_result['complete'] = True
            date_results[analysis_date] = date_result
            continue

        # Check for completion marker
        success_markers = list(current_download_dir.glob(f'download_complete_*.success'))
        partial_markers = list(current_download_dir.glob(f'download_complete_*.partial'))

        if success_markers:
            date_result['completion_file'] = str(max(success_markers, key=lambda p: p.stat().st_mtime))
            logger.info(f"✅ Found success completion marker for {analysis_date}")
        elif partial_markers:
            date_result['completion_file'] = str(max(partial_markers, key=lambda p: p.stat().st_mtime))
            logger.warning(f"⚠️ Found partial completion marker for {analysis_date}")
            date_result['success'] = False
            date_result['complete'] = False
            if strict_mode:
                all_complete = False
                date_result['reason'] = f"Partial downloads: {date_result['failed_downloads']} failed out of {date_result['expected_downloads']}"
                date_results[analysis_date] = date_result
                continue
        else:
            logger.info(f"No completion marker found (download in progress or not started)")
            date_result['success'] = False
            date_result['complete'] = False
            date_result['reason'] = 'No completion marker found (download in progress or not started)'
            all_complete = False
            date_results[analysis_date] = date_result
            missing_dates.append(analysis_date)
            continue

        # Check if any failed downloads
        failed_files = list(current_download_dir.glob('grid_tiles_download_failed_*.txt'))

        if failed_files and date_result['failed_downloads'] > 0:
            failed_grids = []
            for ff in failed_files:
                with open(ff, 'r') as f:
                    failed_grids.extend([line.strip() for line in f.readlines()])

            date_result['failed_grid_tiles'] = failed_grids
            logger.warning(f"Found {len(failed_grids)} failed grid tiles for {analysis_date}")

            if strict_mode:
                date_result['success'] = False
                date_result['complete'] = False
                all_complete = False
                date_result['reason'] = f"{len(failed_grids)} grid tiles failed to download"
                date_results[analysis_date] = date_result
                continue

        # All checks passed for this date
        date_result['complete'] = True
        date_result['success'] = True
        date_result['reason'] = 'All downloads complete and verified'
        date_results[analysis_date] = date_result
        logger.info(f"✅ Date {analysis_date} verification passed")

    # ========== Overall Summary ==========
    logger.info(f"\n{'=' * 80}")
    logger.info("VERIFICATION SUMMARY")
    logger.info(f"{'=' * 80}")

    complete_dates = [d for d, r in date_results.items() if r.get('complete', False)]
    incomplete_dates = [d for d, r in date_results.items() if not r.get('complete', False)]

    logger.info(f"Region: {REGION_NAME}")
    logger.info(f"Total dates verified: {len(date_results)}")
    logger.info(f"Complete dates: {len(complete_dates)}")
    logger.info(f"Incomplete dates: {len(incomplete_dates)}")

    if incomplete_dates:
        for date in incomplete_dates:
            reason = date_results[date].get('reason', 'Unknown reason')
            logger.info(f"  - {date}: {reason}")
    else:
        logger.info("✅ All dates are complete and verified!")

    # Determine if this is a first run
    is_first_run = first_run or len(directory_missing_dates) == len(analysis_dates)

    return {
        'success': True,  # Verification succeeded even if incomplete
        'complete': all_complete if strict_mode else len(incomplete_dates) == 0,
        'first_run': is_first_run,
        'need_download': not all_complete or is_first_run,
        'region': REGION_NAME,
        'dates_verified': analysis_dates,
        'complete_dates': complete_dates,
        'incomplete_dates': incomplete_dates,
        'date_results': date_results,
        'missing_dates': missing_dates,
        'directory_missing_dates': directory_missing_dates,
        'no_manifest_dates': no_manifest_dates,
        'strict_mode': strict_mode,
        'summary': {
            'total_dates': len(date_results),
            'complete_count': len(complete_dates),
            'incomplete_count': len(incomplete_dates),
            'total_expected_downloads': sum(r.get('expected_downloads', 0) for r in date_results.values()),
            'total_successful_downloads': sum(r.get('successful_downloads', 0) for r in date_results.values()),
            'total_failed_downloads': sum(r.get('failed_downloads', 0) for r in date_results.values()),
            'total_skipped_downloads': sum(r.get('skipped_downloads', 0) for r in date_results.values()),
            'directories_missing': len(directory_missing_dates)
        }
    }


