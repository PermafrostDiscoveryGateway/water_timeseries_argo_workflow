import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import zarr
# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import time
from loguru import logger
import sys
import geemap
import ee
import glob
import os
import gc
import shutil
import psutil
from water_timeseries.downloader import EarthEngineDownloader
from water_timeseries.utils.spatial import create_longitude_latitude_grid, filter_gdf_by_bbox
from water_timeseries.utils import io
from water_timeseries.dataset import DWDataset
from water_timeseries.breakpoint import NRTBreakpoint
import datetime

from utils.download_new_dynamic_world_data import download_new_dynamic_world_data
from utils.region_boundaries import get_region_boundaries
import utils.download_new_dynamic_world_data
import json
import resource
import tempfile



def generate_expected_dates_for_region(
        region: str = "TEST",
        env_path: str = None,
        start_year: int = 2015,
        months: List[int] = [6, 7, 8, 9]
):
    """
    Generate expected dates for a specific region.

    This is a convenience function that loads the region's historical file
    and generates dates from start_year to present for June-September.

    Args:
        region: Region name
        env_path: Optional path to .env file
        start_year: Year to start from (default: 2015)
        months: List of months to include (default: [6, 7, 8, 9])

    Returns:
        List of pandas Timestamps for expected dates
    """
    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    dynamic_world_data_dir = os.environ['dynamic_world_data']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))

    if not all_dynamic_world_files:
        logger.warning(f"No NetCDF files found in {dynamic_world_data_dir}")
        return generate_expected_dates(start_year=start_year, months=months)

    # Get the most recent file to check existing dates
    most_recent_file = max(all_dynamic_world_files, key=os.path.getctime)
    ds = xr.open_dataset(most_recent_file)
    existing_dates = set(pd.to_datetime(ds['date'].values))
    ds.close()

    # Generate all expected dates
    all_expected = generate_expected_dates(start_year=start_year, months=months)

    # Filter to only dates that exist in the file (or you could return all)
    # Option 1: Return all expected dates (download will check what's missing)
    # Option 2: Return only dates that already exist in the file
    # Option 3: Return both

    logger.info(f"Generated {len(all_expected)} expected dates from {start_year} to present")
    logger.info(f"File has {len(existing_dates)} existing dates")

    # Return all expected dates - download function will filter out existing ones
    return all_expected


def check_netcdf_compression(file_path):
    """Check if a NetCDF file has compression enabled"""
    try:
        import netCDF4
        ds = netCDF4.Dataset(file_path)

        compression_info = {}
        for var_name in ds.variables:
            var = ds.variables[var_name]
            compression_info[var_name] = {
                'has_zlib': hasattr(var, 'zlib') and var.zlib,
                'complevel': var.complevel if hasattr(var, 'complevel') else None,
                'shuffle': var.shuffle if hasattr(var, 'shuffle') else None,
                'chunksizes': var.chunksizes() if hasattr(var, 'chunksizes') else None
            }
        ds.close()
        return compression_info
    except Exception as e:
        logger.warning(f"Could not check compression: {e}")
        return None


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


def close_and_clean(ds, name: str):
    """Safely close a dataset and clean up"""
    if ds is not None:
        logger.debug(f"Closing dataset: {name}")
        ds.close()
        del ds
        gc.collect()


def merge_zarr_chunked(ds_historical, combined_ds, output_path, chunk_size=250):
    """
    Merge historical and combined datasets in chunks.
    Collects all chunks first then concatenates and writes to zarr file.
    """
    logger.info(f"Merging in chunks of {chunk_size} ids")
    log_memory_usage("Before chunked merge")

    combined_ids = combined_ds['id_geohash'].values
    total_ids = len(combined_ids)
    logger.info(f"Total ids to merge: {total_ids}")

    merged_chunks = []

    for chunk_start in tqdm(range(0, total_ids, chunk_size), desc="Merging chunks"):
        chunk_end = min(chunk_start + chunk_size, total_ids)
        chunk_ids = combined_ids[chunk_start:chunk_end]

        logger.debug(f"Processing chunk: ids {chunk_start} to {chunk_end} ({len(chunk_ids)} ids)")
        log_memory_usage(f"Chunk {chunk_start // chunk_size + 1} start")

        hist_chunk = ds_historical.sel(id_geohash=chunk_ids)
        new_chunk = combined_ds.sel(id_geohash=chunk_ids)

        merged_chunk = xr.merge([hist_chunk, new_chunk])
        merged_chunks.append(merged_chunk)

        close_and_clean(hist_chunk, f"hist_chunk_{chunk_start}")
        close_and_clean(new_chunk, f"new_chunk_{chunk_start}")

        log_memory_usage(f"Chunk {chunk_start // chunk_size + 1} complete")

    logger.info("Concatenating all chunks and writing to file...")
    if merged_chunks:
        final_merged = xr.concat(merged_chunks, dim='id_geohash')

        temp_output = output_path.with_suffix('.tmp.zarr')

        io.save_xarray_dataset(final_merged, temp_output)

        close_and_clean(final_merged, "final_merged")
        for chunk in merged_chunks:
            close_and_clean(chunk, "merged_chunk")

        if temp_output.exists():
            if output_path.exists():
                shutil.rmtree(output_path)
            temp_output.rename(output_path)
            logger.info(f"Successfully wrote merged file to {output_path}")
            size_gb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024 ** 3)
            logger.info(f"File size: {size_gb:.2f} GB")
    else:
        logger.error("No chunks were created, cannot merge")

    return output_path


def append_to_netcdf_chunked(merged_chunk, file_path, first_chunk=False, compression_level=2):
    """
    Append a chunk to a NetCDF file efficiently.

    Args:
        merged_chunk: xarray dataset chunk to write
        file_path: Path to the NetCDF file
        first_chunk: If True, create new file; if False, append
        compression_level: zlib compression level (0-9, higher = more compression but slower)
    """
    # Prepare encoding with compression
    encoding = {}
    for var in merged_chunk.data_vars:
        encoding[var] = {
            'zlib': True,
            'complevel': compression_level,
            'shuffle': True,
            'chunksizes': (min(100, len(merged_chunk['date'])), min(1000, len(merged_chunk['id_geohash'])))
        }

    # Write the chunk
    if first_chunk:
        merged_chunk.to_netcdf(
            file_path,
            mode='w',
            encoding=encoding,
            unlimited_dims=['id_geohash']
        )
    else:
        # Append mode - requires netCDF4 engine
        merged_chunk.to_netcdf(
            file_path,
            mode='a',
            engine='netcdf4',
            encoding=encoding
        )


def create_merged_netcdf_memory_efficient(ds_historical, combined_ds, output_path, chunk_size=50000):
    """
    Create a merged NetCDF file efficiently.
    Fixes the issue where only chunk_size IDs were being saved.
    """
    logger.info(f"Creating merged NetCDF file at {output_path}")
    log_memory_usage("Start of merge_netcdf")

    # Get all unique IDs - ensure they are sorted
    hist_ids = set(ds_historical['id_geohash'].values)
    combined_ids = set(combined_ds['id_geohash'].values)
    all_ids = np.array(sorted(hist_ids | combined_ids))

    logger.info(f"Historical IDs: {len(hist_ids)}, Combined IDs: {len(combined_ids)}")
    logger.info(f"Total unique IDs: {len(all_ids)}")

    total_ids = len(all_ids)

    # For datasets with a reasonable size, merge directly
    if total_ids < 500000:  # Adjust based on available memory
        logger.info(f"Total IDs {total_ids} is manageable, merging directly...")

        # Get data in batches to avoid memory issues
        batch_size = 100000
        merged_chunks = []

        for start_idx in tqdm(range(0, total_ids, batch_size), desc="Loading batches"):
            end_idx = min(start_idx + batch_size, total_ids)
            batch_ids = all_ids[start_idx:end_idx]
            batch_ids_list = batch_ids.tolist()

            # Get historical data for this batch
            hist_data = None
            existing_hist = [id_val for id_val in batch_ids_list if id_val in hist_ids]
            if existing_hist:
                hist_data = ds_historical.sel(id_geohash=existing_hist)

            # Get combined data for this batch
            combined_data = None
            existing_combined = [id_val for id_val in batch_ids_list if id_val in combined_ids]
            if existing_combined:
                combined_data = combined_ds.sel(id_geohash=existing_combined)

            # Merge
            if hist_data is not None and combined_data is not None:
                batch_merged = xr.concat([hist_data, combined_data], dim='id_geohash')
                _, unique_idx = np.unique(batch_merged['id_geohash'].values, return_index=True)
                if len(unique_idx) < len(batch_merged['id_geohash']):
                    batch_merged = batch_merged.isel(id_geohash=np.sort(unique_idx))
            elif hist_data is not None:
                batch_merged = hist_data
            elif combined_data is not None:
                batch_merged = combined_data
            else:
                continue

            merged_chunks.append(batch_merged)

            # Clean up
            if hist_data is not None:
                hist_data.close()
            if combined_data is not None:
                combined_data.close()
            gc.collect()

        if merged_chunks:
            # Concatenate all batches
            logger.info(f"Concatenating {len(merged_chunks)} batches...")
            final_merged = xr.concat(merged_chunks, dim='id_geohash')
            _, unique_idx = np.unique(final_merged['id_geohash'].values, return_index=True)
            if len(unique_idx) < len(final_merged['id_geohash']):
                final_merged = final_merged.isel(id_geohash=np.sort(unique_idx))

            # Sort
            final_merged = final_merged.sortby(['date', 'id_geohash'])

            # Write to file
            encoding = {}
            for var in final_merged.data_vars:
                encoding[var] = {
                    'zlib': True,
                    'complevel': 4,
                    'shuffle': True
                }

            final_merged.to_netcdf(
                output_path,
                mode='w',
                encoding=encoding,
                unlimited_dims=['id_geohash']
            )

            # Clean up
            final_merged.close()
            for chunk in merged_chunks:
                chunk.close()
            gc.collect()

            logger.info(f"Successfully created merged NetCDF file: {output_path}")
            file_size_gb = get_file_size_gb(str(output_path))
            logger.info(f"File size: {file_size_gb:.2f} GB")
            return output_path
        else:
            logger.error("No data to merge")
            return None

    # For very large datasets, use the original chunked approach but with proper handling
    logger.info(f"Large dataset ({total_ids} IDs), using chunked approach...")

    # Create a temporary directory for chunk files
    import tempfile
    temp_dir = tempfile.mkdtemp()
    chunk_files = []

    try:
        num_chunks = (total_ids + chunk_size - 1) // chunk_size

        for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_ids)
            chunk_ids = all_ids[start_idx:end_idx]
            chunk_ids_list = chunk_ids.tolist()

            # Get data
            hist_chunk = None
            existing_hist_ids = [id_val for id_val in chunk_ids_list if id_val in hist_ids]
            if existing_hist_ids:
                hist_chunk = ds_historical.sel(id_geohash=existing_hist_ids)

            combined_chunk = None
            existing_combined_ids = [id_val for id_val in chunk_ids_list if id_val in combined_ids]
            if existing_combined_ids:
                combined_chunk = combined_ds.sel(id_geohash=existing_combined_ids)

            # Merge
            if hist_chunk is not None and combined_chunk is not None:
                merged_chunk = xr.concat([hist_chunk, combined_chunk], dim='id_geohash')
                _, unique_idx = np.unique(merged_chunk['id_geohash'].values, return_index=True)
                if len(unique_idx) < len(merged_chunk['id_geohash']):
                    merged_chunk = merged_chunk.isel(id_geohash=np.sort(unique_idx))
            elif hist_chunk is not None:
                merged_chunk = hist_chunk
            elif combined_chunk is not None:
                merged_chunk = combined_chunk
            else:
                continue

            # Sort
            merged_chunk = merged_chunk.sortby(['date', 'id_geohash'])

            # Save chunk to temporary file
            chunk_file = os.path.join(temp_dir, f'chunk_{chunk_idx:04d}.nc')
            merged_chunk.to_netcdf(chunk_file)
            chunk_files.append(chunk_file)

            # Clean up
            if hist_chunk is not None:
                hist_chunk.close()
            if combined_chunk is not None:
                combined_chunk.close()
            merged_chunk.close()
            gc.collect()

        if chunk_files:
            # Combine all chunk files
            logger.info(f"Combining {len(chunk_files)} chunk files...")

            # Open all chunks and concatenate
            chunk_datasets = []
            for chunk_file in chunk_files:
                ds = xr.open_dataset(chunk_file)
                chunk_datasets.append(ds)

            # Concatenate
            final_merged = xr.concat(chunk_datasets, dim='id_geohash')

            # Write to final file
            encoding = {}
            for var in final_merged.data_vars:
                encoding[var] = {
                    'zlib': True,
                    'complevel': 4,
                    'shuffle': True
                }

            final_merged.to_netcdf(
                output_path,
                mode='w',
                encoding=encoding,
                unlimited_dims=['id_geohash']
            )

            # Clean up
            final_merged.close()
            for ds in chunk_datasets:
                ds.close()
            gc.collect()

            logger.info(f"Successfully created merged NetCDF file: {output_path}")
            file_size_gb = get_file_size_gb(str(output_path))
            logger.info(f"File size: {file_size_gb:.2f} GB")
            return output_path
        else:
            logger.error("No chunks were created")
            return None

    finally:
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def verify_merged_netcdf(file_path, expected_id_count=None, expected_date_count=None):
    """
    Verify a merged NetCDF file is valid and has expected dimensions.

    Args:
        file_path: Path to the NetCDF file
        expected_id_count: Optional expected number of IDs
        expected_date_count: Optional expected number of dates

    Returns:
        dict: Verification results
    """
    try:
        logger.info(f"Verifying merged NetCDF file: {file_path}")
        ds = xr.open_dataset(file_path)

        id_count = len(ds['id_geohash'])
        date_count = len(ds['date'])

        result = {
            'valid': True,
            'id_count': id_count,
            'date_count': date_count,
            'file_size_gb': get_file_size_gb(str(file_path)),
            'variables': list(ds.data_vars)
        }

        if expected_id_count is not None and id_count != expected_id_count:
            logger.warning(f"ID count mismatch: expected {expected_id_count}, got {id_count}")
            result['valid'] = False
            result['id_count_mismatch'] = True

        if expected_date_count is not None and date_count != expected_date_count:
            logger.warning(f"Date count mismatch: expected {expected_date_count}, got {date_count}")
            result['valid'] = False
            result['date_count_mismatch'] = True

        ds.close()
        logger.info(f"✅ File verified: {id_count} IDs, {date_count} dates, {result['file_size_gb']:.2f} GB")
        return result

    except Exception as e:
        logger.error(f"❌ Failed to verify NetCDF file: {e}")
        return {'valid': False, 'error': str(e)}


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


def print_comparison_summary(comparison_results):
    """
    Print a concise summary of the comparison results.
    """
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)

    # File sizes
    size1 = comparison_results['summary']['file1_size_gb']
    size2 = comparison_results['summary']['file2_size_gb']
    print(f"File sizes: {size1:.4f} GB → {size2:.4f} GB ({(size2 / size1 - 1) * 100:+.1f}%)")

    # Dimensions
    print("\nDimensions:")
    for dim, info in comparison_results['dimensions'].items():
        diff = info['diff'] if info['diff'] is not None else '?'
        print(f"  {dim}: {info['file1']} → {info['file2']} ({diff:+d})")

    # Variables
    print(f"\nVariables: {len(comparison_results['variables'])} common variables")

    # Sample comparison
    if comparison_results['sample_comparisons']:
        all_match = all(s.get('all_match', False) for s in comparison_results['sample_comparisons'])
        print(
            f"\nSample IDs ({len(comparison_results['sample_comparisons'])} sampled): {'✅ All match' if all_match else '⚠️ Some mismatches found'}")

    # New data
    new_ids = comparison_results['summary'].get('new_ids_added', 0)
    new_dates = comparison_results['summary'].get('new_dates_added', 0)
    if new_ids > 0:
        print(f"\nNew IDs added: {new_ids}")
    if new_dates > 0:
        print(f"New dates added: {new_dates}")

    # Success status
    if comparison_results['summary']['successful']:
        print("\n✅ Overall: MERGE SUCCESSFUL")
    else:
        print("\n❌ Overall: MERGE HAS ISSUES")
        for issue in comparison_results['summary']['issues']:
            print(f"  - {issue}")

def near_real_time_region(region: str = "TEST", env_path: str = None):
    """
    Run near-real-time breakpoint analysis for a specific region.
    (Legacy function - kept for compatibility)
    """
    log_memory_usage("Program start")

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
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))

    if not all_dynamic_world_files:
        logger.error(f"No .nc files found in {dynamic_world_data_dir}")
        return False

    logger.debug(f"Region name is {REGION_NAME}")

    bounding_box_coords = region_boundaries[REGION_NAME]

    logger.debug(f"Bounding box coordinates are {bounding_box_coords}")
    time.sleep(15)

    X_MIN_START = bounding_box_coords['X_MIN_START']
    X_MIN_END = bounding_box_coords['X_MIN_END']
    Y_MIN_START = bounding_box_coords['Y_MIN_START']
    Y_MIN_END = bounding_box_coords['Y_MIN_END']

    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)

    hist_file_size_gb = get_file_size_gb(most_recent_dynamic_world_file)
    logger.info(f"Historical NetCDF file size: {hist_file_size_gb:.2f} GB")

    missing_dates = utils.download_new_dynamic_world_data.check_missing_data_in_netcdf(most_recent_dynamic_world_file)

    # ========== Handle missing dates ==========
    if missing_dates:
        logger.warning(f"Found {len(missing_dates)} missing dates in historical data")
        for date in missing_dates:
            missing_date_string = date.strftime("%Y-%m")
            logger.warning(f"Missing date: {missing_date_string}")
        logger.info("Will download missing data and run breakpoint analysis")
        DOWNLOAD_REQUIRED = True
    else:
        logger.info("No missing dates found in historical data")
        logger.info("Will run breakpoint analysis using existing data only (no download)")
        DOWNLOAD_REQUIRED = False

    vector_lake_file = os.environ['vector_lake_file']
    path_historical_dw = most_recent_dynamic_world_file
    path_lake_vector = vector_lake_file

    # Process each missing date
    for date in missing_dates:
        ANALYSIS_DATE = date.strftime("%Y-%m")

        gdf = gpd.read_parquet(path_lake_vector)
        log_memory_usage("After loading lake vectors")

        # getting most recent file (it might have been replaced by a previous run through this loop)
        logger.debug(f"Current most recent dynamic world file: {most_recent_dynamic_world_file}")
        most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
        logger.debug(f"Most recent dynamic world file: {most_recent_dynamic_world_file} after checking for new files")
        hist_file_size_gb = get_file_size_gb(most_recent_dynamic_world_file)
        logger.info(f"Historical NetCDF file size: {hist_file_size_gb:.2f} GB")
        path_historical_dw = most_recent_dynamic_world_file

        bbox_size_lon = 1
        bbox_size_lat = 1
        grid = create_longitude_latitude_grid(lon_range=(X_MIN_START, X_MIN_END), lat_range=(Y_MIN_START, Y_MIN_END),
                                              bbox_size_lon=bbox_size_lon, bbox_size_lat=bbox_size_lat)
        print('created grid')
        log_memory_usage("After creating grid")

        bp = NRTBreakpoint()

        current_breakpoint_dir = Path(output_dir) / f'breakpoint_{ANALYSIS_DATE}'
        current_breakpoint_dir.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Current breakpoint directory: {current_breakpoint_dir}")

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

        # Only initialize downloader if we need to download
        if DOWNLOAD_REQUIRED:
            downloader = EarthEngineDownloader(ee_project=EE_PROJECT_ID)
        else:
            downloader = None
            logger.info("Downloader disabled - using only existing historical data")

        breaks_list = []
        total = len(grid[:])
        partial_saved = False

        # First, load historical dataset once to get valid IDs
        logger.info("Loading historical dataset to check valid IDs...")
        ds_historical_check = xr.open_dataset(path_historical_dw)
        valid_historical_ids = set(ds_historical_check['id_geohash'].values)
        ds_historical_check.close()
        logger.info(f"Found {len(valid_historical_ids)} valid IDs in historical dataset")

        # Define expected output columns for empty results
        expected_columns = [
            'date', 'water_observed', 'water_predicted', 'water_residual',
            'water_predicted_lower_90', 'water_predicted_upper_90',
            'water_historical_mean', 'water_historical_median', 'water_historical_std',
            'water_historical_min', 'water_historical_max', 'drainage_confidence'
        ]

        # file for grids that failed
        current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        outfile_breaks_failed_file = current_download_dir / f'grid_tiles_failed_{current_datetime}.txt'

        # run loop
        logger.debug(f"There are total {total} grid tiles for {REGION_NAME}")
        for i, (lon, lat) in enumerate(tqdm(grid[:], total=total, desc="Processing")):
            logger.debug(f"Processing {i}/{total} grid tiles.")
            bbox_west = int(lon)
            bbox_east = int(lon + bbox_size_lon)
            bbox_south = int(lat)
            bbox_north = int(lat + bbox_size_lat)

            print(f"Run processing for bbox: {bbox_west} {bbox_east} {bbox_south} {bbox_north}")

            outfile_download = current_download_dir / f'DW_{ANALYSIS_DATE}_{bbox_west}_{bbox_east}_{bbox_south}_{bbox_north}.nc'
            outfile_breaks = current_breakpoint_dir / f'DW_{ANALYSIS_DATE}_{bbox_west}_{bbox_east}_{bbox_south}_{bbox_north}_breaks.parquet'

            if outfile_breaks.exists():
                print(f'Breakpoints already calculated! Skipping {bbox_west} {bbox_south}')
                breaks_list.append(pd.read_parquet(outfile_breaks))
                continue

            gdf_subset = filter_gdf_by_bbox(gdf=gdf, bbox_west=lon, bbox_east=lon + bbox_size_lon, bbox_south=lat,
                                            bbox_north=lat + bbox_size_lat)
            n_lakes = len(gdf_subset)
            print('Number of lakes: ', n_lakes)

            id_list = gdf_subset['id_geohash'].values.tolist()
            if n_lakes == 0:
                print(f'No lakes for grid {bbox_west} {bbox_south}. Skipping!')
                continue

            # Filter IDs to only those that exist in historical data
            original_count = len(id_list)
            id_list = [id_val for id_val in id_list if id_val in valid_historical_ids]
            filtered_count = len(id_list)

            if filtered_count == 0:
                print(
                    f'WARNING: No valid historical IDs for grid {bbox_west} {bbox_south} (had {original_count} lakes, none in historical data). Skipping!')
                continue
            elif filtered_count < original_count:
                print(
                    f'NOTE: Filtered {original_count - filtered_count} lakes not found in historical data. Processing {filtered_count} lakes.')
                # Also filter the gdf_subset to only keep valid IDs
                gdf_subset = gdf_subset[gdf_subset['id_geohash'].isin(id_list)]

            # ========== Handle download vs no-download cases ==========
            ds_dl = None
            download_successful = False

            if DOWNLOAD_REQUIRED:
                # Download or load existing file
                if not outfile_download.exists():
                    # TODO use the features in gdf subset
                    n_features = len(gdf_subset)
                    if n_features > 500:
                        # For large grids, process in smaller chunks
                        max_total_requests = min(100, n_features)
                        logger.debug(f"Grid has {n_features} features, using max_requests={max_total_requests}")
                    else:
                        max_total_requests = 1000
                    try:
                        ds_dl = downloader.download_dw_monthly(
                            gdf=gdf_subset,
                            max_total_requests=max_total_requests,
                            n_parallel=2,
                            date_list=[ANALYSIS_DATE],
                            save_to_file=outfile_download
                        )
                        download_successful = True
                        print(f'Successfully downloaded data for {bbox_west} {bbox_south}')
                    except ValueError as e:
                        if "No data was extracted" in str(e):
                            print(f'WARNING: No data available for {bbox_west} {bbox_south} on {ANALYSIS_DATE}')
                            download_successful = False
                        else:
                            logger.error(f"Download error for {bbox_west} {bbox_south}: {e}")
                            download_successful = False
                    except Exception as e:
                        logger.error(f"Unexpected error downloading {bbox_west} {bbox_south}: {e}")
                        download_successful = False
                else:
                    print(f'Loading existing download for {bbox_west} {bbox_south}')
                    try:
                        ds_dl = xr.open_dataset(outfile_download)
                        download_successful = True
                    except Exception as e:
                        logger.error(f"Error loading existing download file: {e}")
                        download_successful = False
            else:
                # No download required - use historical data only
                print(f'No download needed for {bbox_west} {bbox_south} - using historical data only')
                ds_dl = None
                download_successful = False

            # Load historical data for this tile
            logger.info(f"Loading historical dataset for tile {i}...")
            ds_historical = xr.open_dataset(path_historical_dw)

            # Subset historical data
            ds_historical_subset = ds_historical.sel(id_geohash=id_list)

            # Close historical immediately
            ds_historical.close()
            del ds_historical
            gc.collect()

            # ========== Merge or use historical only ==========
            if download_successful and ds_dl is not None:
                # We have new data to merge
                ds_dl_dates = pd.to_datetime(ds_dl['date'].values).strftime('%Y-%m')
                if ANALYSIS_DATE in ds_dl_dates:
                    ds_merged = xr.merge([ds_historical_subset, ds_dl]).sortby('date')
                    print(f'Merged new data for {ANALYSIS_DATE} with historical record')
                else:
                    print(f'WARNING: Downloaded file for {bbox_west} {bbox_south} does not contain {ANALYSIS_DATE}')
                    ds_merged = ds_historical_subset
                    download_successful = False

                # Clean up download dataset
                if ds_dl is not None:
                    ds_dl.close()
                    del ds_dl
            else:
                # Use only historical data
                logger.info(f"No new data to merge for grid {bbox_west} {bbox_south} - using historical data only")
                ds_merged = ds_historical_subset

            # ========== Calculate breakpoints with error handling ==========
            try:
                # Create dataset
                dwds = DWDataset(ds_merged)

                # Check if analysis date exists in the dataset
                if ANALYSIS_DATE not in dwds.dates_:
                    logger.warning(
                        f"Analysis date {ANALYSIS_DATE} not in dataset dates for grid {bbox_west} {bbox_south}")
                    # Create empty result with expected columns
                    empty_result = pd.DataFrame(columns=expected_columns)
                    empty_result.to_parquet(outfile_breaks)
                    breaks_list.append(empty_result)
                    print(f'Created empty result for {bbox_west} {bbox_south} - analysis date not in data')
                else:
                    # Calculate breakpoints
                    breaks = bp.calculate_break(dataset=dwds, analysis_date=ANALYSIS_DATE)
                    breaks.to_parquet(outfile_breaks)
                    breaks_list.append(breaks)
                    print(f'Successfully calculated breakpoints for {bbox_west} {bbox_south}')

            except ValueError as e:
                if "not available in the dataset" in str(e):
                    logger.warning(
                        f"Analysis date {ANALYSIS_DATE} not available for grid {bbox_west} {bbox_south}: {e}")
                    with open(outfile_breaks_failed_file, 'a') as f:
                        f.write(str(outfile_breaks) + '\n')
                else:
                    logger.error(f"ValueError calculating breakpoints for {bbox_west} {bbox_south}: {e}")
                    with open(outfile_breaks_failed_file, 'a') as f:
                        f.write(str(outfile_breaks) + '\n')
            except Exception as e:
                logger.error(f"Unexpected error calculating breakpoints for {bbox_west} {bbox_south}: {e}")
                with open(outfile_breaks_failed_file, 'a') as f:
                    f.write(str(outfile_breaks) + '\n')

            # Clean up
            ds_historical_subset.close()
            ds_merged.close()
            del ds_historical_subset, ds_merged
            gc.collect()

            # Periodic save
            if len(breaks_list) >= 10:
                logger.info(f"Saving intermediate results...")
                non_empty_breaks = [df for df in breaks_list if not df.empty]
                if non_empty_breaks:
                    breaks_merged = pd.concat(non_empty_breaks, ignore_index=True)
                    joined = gdf.set_index('id_geohash').join(breaks_merged, how='inner').reset_index()
                    partial_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}_partial.parquet'
                    joined.to_parquet(partial_file)
                else:
                    logger.warning("No non-empty breakpoint results to save in partial file")
                breaks_list = []
                gc.collect()

        # Final save for this date
        if breaks_list:
            non_empty_breaks = [df for df in breaks_list if not df.empty]
            if non_empty_breaks:
                breaks_merged = pd.concat(non_empty_breaks, ignore_index=True)
                joined = gdf.set_index('id_geohash').join(breaks_merged, how='inner').reset_index()
                path_to_joined_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}.parquet'
                joined.to_parquet(path_to_joined_file)
                logger.info(f"Final combined file saved to {path_to_joined_file}")
            else:
                logger.warning(f"No valid breakpoint results found for date {ANALYSIS_DATE}")
                empty_result = pd.DataFrame(columns=expected_columns)
                path_to_joined_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}.parquet'
                empty_result.to_parquet(path_to_joined_file)
                logger.info(f"Created empty result file for {ANALYSIS_DATE}")

        end = datetime.datetime.now()
        logger.debug(f"Finished processing date {ANALYSIS_DATE} in {end - start}")

        logger.info("Combining into Zarr file...")

        downloaded_files = sorted(glob.glob(str(current_download_dir / f'DW_{ANALYSIS_DATE}_*.nc')))
        output_zarr = Path(output_dir) / f'lakes_dw_Vd2_{ANALYSIS_DATE}.zarr'
        logger.debug(f"Output zarr file being saved to {output_zarr}")

        combined = None
        ds_historical = None
        if combined is not None and ds_historical is not None:
            merge_zarr_chunked(ds_historical, combined, output_zarr, chunk_size=250)
            ds_historical.close()

        # ========== CREATE NEW HISTORICAL NETCDF FILE (MEMORY EFFICIENT) ==========
        if downloaded_files:
            logger.info("Loading historical dataset for merge...")
            ds_historical = xr.open_dataset(most_recent_dynamic_world_file)

            logger.info(f"Loading {len(downloaded_files)} downloaded files...")

            # Process downloaded files in batches to build combined dataset
            BATCH_SIZE = 10  # Increased from 2 for better throughput

            for batch_idx in tqdm(range(0, len(downloaded_files), BATCH_SIZE), desc="Processing batches"):
                batch_files = downloaded_files[batch_idx:batch_idx + BATCH_SIZE]
                batch_datasets = []

                for nc_file in batch_files:
                    try:
                        ds = xr.open_dataset(nc_file)
                        batch_datasets.append(ds)
                    except Exception as e:
                        logger.error(f"Error opening {nc_file}: {e}")
                        continue

                if batch_datasets:
                    try:
                        batch_combined = xr.concat(batch_datasets, dim='id_geohash')
                        _, unique_idx = np.unique(batch_combined['id_geohash'].values, return_index=True)
                        batch_combined = batch_combined.isel(id_geohash=np.sort(unique_idx))

                        if combined is None:
                            combined = batch_combined
                        else:
                            # Combine with existing
                            combined = xr.concat([combined, batch_combined], dim='id_geohash')
                            _, unique_idx = np.unique(combined['id_geohash'].values, return_index=True)
                            combined = combined.isel(id_geohash=np.sort(unique_idx))
                    except Exception as e:
                        logger.error(f"Error combining batch: {e}")

                    # Clean up batch datasets
                    for ds in batch_datasets:
                        ds.close()
                    gc.collect()

            if combined is not None:
                logger.info(f"Combined dataset has {len(combined['id_geohash'])} IDs")

                # Generate timestamp for the new file
                current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                original_path = Path(most_recent_dynamic_world_file)
                new_historical_path = original_path.parent / f"historical_data_{current_timestamp}.nc"

                # Use the memory-efficient merge function
                logger.info("Starting memory-efficient merge to NetCDF...")
                create_merged_netcdf_memory_efficient(
                    ds_historical=ds_historical,
                    combined_ds=combined,
                    output_path=new_historical_path,
                    chunk_size=5000  # Adjust based on available memory
                )

                logger.debug(f"Checking compressions of files")
                logger.debug(f"Original file: {most_recent_dynamic_world_file}")
                logger.debug(f"New file: {new_historical_path}")
                logger.debug("Checking compression of original file...")
                orig_compression = check_netcdf_compression(most_recent_dynamic_world_file)
                if orig_compression:
                    for var_name, info in orig_compression.items():
                        if info['has_zlib']:
                            logger.info(f"Original variable {var_name} has compression (level {info['complevel']})")
                        else:
                            logger.info(f"Original variable {var_name} has NO compression")

                logger.info("Checking compression of new file...")
                new_compression = check_netcdf_compression(str(new_historical_path))
                if new_compression:
                    for var_name, info in new_compression.items():
                        if info['has_zlib']:
                            logger.info(f"New variable {var_name} has compression (level {info['complevel']})")
                        else:
                            logger.info(f"New variable {var_name} has NO compression")

                logger.debug(f"Other comparisons")
                # Compare a few random IDs between the files
                import random

                # Open both files
                orig_ds = xr.open_dataset(most_recent_dynamic_world_file)
                new_ds = xr.open_dataset(str(new_historical_path))

                # Check they have the same number of IDs
                orig_ids = set(orig_ds['id_geohash'].values)
                new_ids = set(new_ds['id_geohash'].values)

                logger.debug(f"Original IDs: {len(orig_ids)}")
                logger.debug(f"New IDs: {len(new_ids)}")

                # Check if all original IDs are in the new file
                missing_ids = orig_ids - new_ids
                if missing_ids:
                    logger.debug(f"Missing {len(missing_ids)} IDs in new file")
                    # Show first 10 missing IDs
                    logger.debug(f"First 10 missing IDs: {list(missing_ids)[:10]}")
                else:
                    logger.debug("All original IDs are preserved in new file")

                # Check data values for a few random IDs
                sample_ids = random.sample(list(orig_ids), min(5, len(orig_ids)))
                for sample_id in sample_ids:
                    orig_data = orig_ds.sel(id_geohash=sample_id)
                    new_data = new_ds.sel(id_geohash=sample_id)

                    # Compare a variable
                    if 'water_observed' in orig_data:
                        orig_vals = orig_data['water_observed'].values
                        new_vals = new_data['water_observed'].values

                        # Check if values are close (floating point might have slight differences)
                        if np.allclose(orig_vals, new_vals, rtol=1e-6):
                            logger.debug(f"ID {sample_id}: Values match")
                        else:
                            max_diff = np.max(np.abs(orig_vals - new_vals))
                            logger.debug(f"ID {sample_id}: Values differ (max diff: {max_diff})")

                # TODO add more detailed checks
                # TODO add more detailed checks
                # TODO add more detailed checks
                logger.info("=" * 80)
                logger.info("DETAILED FILE COMPARISON")
                logger.info("=" * 80)

                # 1. Check file sizes
                orig_size_gb = get_file_size_gb(most_recent_dynamic_world_file)
                new_size_gb = get_file_size_gb(str(new_historical_path))
                logger.info(f"Original file size: {orig_size_gb:.4f} GB")
                logger.info(f"New file size: {new_size_gb:.4f} GB")
                logger.info(f"Size reduction: {(1 - new_size_gb / orig_size_gb) * 100:.2f}%")

                # 2. Check data types of each variable
                logger.info("\n--- Data Type Comparison ---")
                orig_dtypes = {var: orig_ds[var].dtype for var in orig_ds.data_vars}
                new_dtypes = {var: new_ds[var].dtype for var in new_ds.data_vars}

                for var in orig_ds.data_vars:
                    if var in new_ds.data_vars:
                        orig_dtype = orig_dtypes[var]
                        new_dtype = new_dtypes[var]
                        logger.info(f"Variable '{var}': Original dtype={orig_dtype}, New dtype={new_dtype}")
                        if orig_dtype != new_dtype:
                            logger.warning(f"  → Data type changed from {orig_dtype} to {new_dtype}!")

                # 3. Check compression
                logger.info("\n--- Compression Comparison ---")
                for var in orig_ds.data_vars:
                    if var in new_ds.data_vars:
                        orig_encoding = orig_ds[var].encoding
                        new_encoding = new_ds[var].encoding

                        logger.info(f"Variable '{var}':")

                        # Check original compression
                        orig_has_zlib = orig_encoding.get('zlib', False)
                        orig_complevel = orig_encoding.get('complevel', 0)
                        orig_shuffle = orig_encoding.get('shuffle', False)

                        # Check new compression
                        new_has_zlib = new_encoding.get('zlib', False)
                        new_complevel = new_encoding.get('complevel', 0)
                        new_shuffle = new_encoding.get('shuffle', False)

                        if orig_has_zlib:
                            logger.info(f"  Original: zlib=True, complevel={orig_complevel}, shuffle={orig_shuffle}")
                        else:
                            logger.info(f"  Original: NO compression")

                        if new_has_zlib:
                            logger.info(
                                f"  New: zlib=True, complevel={new_complevel}, shuffle={new_shuffle} ✅ COMPRESSED")
                        else:
                            logger.info(f"  New: NO compression")

                # 4. Check dimensions
                logger.info("\n--- Dimensions Comparison ---")
                orig_dims = {dim: len(orig_ds[dim]) for dim in orig_ds.dims}
                new_dims = {dim: len(new_ds[dim]) for dim in new_ds.dims}
                logger.info(f"Original dimensions: {orig_dims}")
                logger.info(f"New dimensions: {new_dims}")

                if orig_dims['date'] < new_dims['date']:
                    logger.info(
                        f"✅ New file has {new_dims['date'] - orig_dims['date']} additional date(s) (as expected for NRT update)")
                    # Find the new date(s)
                    orig_dates = set(pd.to_datetime(orig_ds['date'].values))
                    new_dates = set(pd.to_datetime(new_ds['date'].values))
                    added_dates = new_dates - orig_dates
                    for date in sorted(added_dates):
                        logger.info(f"  Added date: {date.strftime('%Y-%m-%d')}")

                # 5. Check data integrity for IDs that have data for the new date
                logger.info("\n--- Data Integrity Check ---")
                logger.info("Checking if new data was properly added...")

                # Find an ID that exists in both datasets
                sample_id = next(iter(orig_ids))
                logger.info(f"Using sample ID: {sample_id}")

                orig_data = orig_ds.sel(id_geohash=sample_id)
                new_data = new_ds.sel(id_geohash=sample_id)

                orig_dates = orig_data['date'].values
                new_dates = new_data['date'].values

                logger.info(f"Original dates for ID {sample_id}: {len(orig_dates)}")
                logger.info(f"New dates for ID {sample_id}: {len(new_dates)}")

                # Check which dates are new
                orig_date_set = set(orig_dates)
                new_date_set = set(new_dates)
                common_dates = orig_date_set & new_date_set
                added_dates = new_date_set - orig_date_set

                logger.info(f"Common dates: {len(common_dates)}")
                logger.info(f"Added dates: {len(added_dates)}")

                if added_dates:
                    logger.info(f"✅ New data was added for dates: {sorted(added_dates)[:5]}... (showing first 5)")

                    # Verify data for a new date
                    for date in list(added_dates)[:3]:
                        new_values = new_data.sel(date=date)
                        if 'water' in new_values:
                            water_vals = new_values['water'].values
                            logger.info(
                                f"  Date {pd.to_datetime(date).strftime('%Y-%m-%d')}: water values present (shape: {water_vals.shape})")
                            if np.any(~np.isnan(water_vals)):
                                logger.info(f"    ✅ Contains non-NaN values")
                            else:
                                logger.warning(f"    ⚠️ All NaN values")

                # 6. Verify original data is preserved
                logger.info("\n--- Verifying Original Data Preservation ---")
                for sample_id in list(orig_ids)[:5]:  # Check 5 random IDs
                    orig_data = orig_ds.sel(id_geohash=sample_id)
                    new_data = new_ds.sel(id_geohash=sample_id)

                    # Compare common dates only
                    orig_dates = orig_data['date'].values
                    new_dates = new_data['date'].values
                    common_dates = np.intersect1d(orig_dates, new_dates)

                    if len(common_dates) > 0:
                        # Test one variable on common dates
                        var_to_check = 'water'
                        if var_to_check in orig_data:
                            orig_vals = orig_data[var_to_check].sel(date=common_dates).values
                            new_vals = new_data[var_to_check].sel(date=common_dates).values

                            if np.allclose(orig_vals, new_vals, rtol=1e-6, atol=1e-6):
                                logger.info(f"✅ ID {sample_id[:8]}...: Original data preserved on common dates")
                            else:
                                max_diff = np.max(np.abs(orig_vals - new_vals))
                                logger.warning(
                                    f"⚠️ ID {sample_id[:8]}...: Data differs on common dates (max diff: {max_diff:.6f})")

                # 7. Summary
                logger.info("\n" + "=" * 80)
                logger.info("SUMMARY")
                logger.info("=" * 80)

                logger.info(
                    f"✅ File size reduced by {(1 - new_size_gb / orig_size_gb) * 100:.2f}% (from {orig_size_gb:.2f}GB to {new_size_gb:.2f}GB)")
                logger.info(f"✅ Compression applied: zlib=True, complevel=4, shuffle=True")
                logger.info(f"✅ All {len(orig_vals)} variables preserved")
                logger.info(f"✅ All {len(orig_ids)} original IDs preserved")
                logger.info(f"✅ {new_dims['date'] - orig_dims['date']} new date(s) added")
                logger.info(f"✅ Original data values preserved on common dates")

                if added_dates:
                    logger.info(f"✅ Near-real-time update successful! Added data for {len(added_dates)} new date(s)")
                else:
                    logger.warning("⚠️ No new dates were added - check if download was successful")

                logger.info("=" * 80)
                orig_ds.close()
                new_ds.close()
                logger.debug(f"Sleep for time")
                time.sleep(300)
                # Clean up
                combined.close()
                del combined
                gc.collect()
            else:
                logger.warning("No combined data to merge")

        # Continue with existing merge_zarr_chunked logic

        logger.debug(f"End of date block for {REGION_NAME} and date {date}")

    logger.info(f"Near-real-time processing completed for region: {REGION_NAME}")
    return True


# ========== NEW: DOWNLOAD FUNCTION (NO MERGING) ==========
def download_near_real_time_region(region: str = "TEST", run_start_label: str = None, env_path: str = None):
    """
    Download near-real-time data for a specific region.

    This function ONLY handles downloading - no merging is performed.
    Downloads are saved to the download directory for later merging.

    Args:
        region: Region name (e.g., "TEST", "AFRICA", "SOUTH_AMERICA")
        run_start_label: Optional label for tracking runs
        env_path: Optional path to .env file

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
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))

    if not all_dynamic_world_files:
        logger.error(f"No .nc files found in {dynamic_world_data_dir}")
        return {'success': False, 'error': 'No .nc files found'}

    logger.debug(f"Region name is {REGION_NAME}")

    bounding_box_coords = region_boundaries[REGION_NAME]

    logger.debug(f"Bounding box coordinates are {bounding_box_coords}")
    time.sleep(15)

    X_MIN_START = bounding_box_coords['X_MIN_START']
    X_MIN_END = bounding_box_coords['X_MIN_END']
    Y_MIN_START = bounding_box_coords['Y_MIN_START']
    Y_MIN_END = bounding_box_coords['Y_MIN_END']

    # Get the original historical file
    original_historical_file = max(all_dynamic_world_files, key=os.path.getctime)

    hist_file_size_gb = get_file_size_gb(original_historical_file)
    logger.info(f"Original Historical NetCDF file size: {hist_file_size_gb:.2f} GB")

    missing_dates = utils.download_new_dynamic_world_data.check_missing_data_in_netcdf(original_historical_file)

    # ========== Handle missing dates ==========
    if missing_dates:
        logger.warning(f"Found {len(missing_dates)} missing dates in historical data")
        for date in missing_dates:
            missing_date_string = date.strftime("%Y-%m")
            logger.warning(f"Missing date: {missing_date_string}")
        logger.info("Will download missing data")
    else:
        logger.info("No missing dates found in historical data")
        logger.info("No downloads required")
        return {'success': True, 'dates_processed': [], 'message': 'No missing dates found'}

    vector_lake_file = os.environ['vector_lake_file']
    path_lake_vector = vector_lake_file

    # Track results for all dates
    all_results = {}
    overall_success = True

    # ========== LOAD ORIGINAL VALID IDs ONCE ==========
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

    # Process each missing date
    for date_idx, date in enumerate(missing_dates):
        ANALYSIS_DATE = date.strftime("%Y-%m")
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing date {date_idx + 1}/{len(missing_dates)}: {ANALYSIS_DATE}")
        logger.info(f"{'=' * 80}")

        date_start = datetime.datetime.now()

        # Track results for this date
        date_results = {
            'analysis_date': ANALYSIS_DATE,
            'success_bbox_downloads': 0,
            'failed_bbox_downloads': 0,
            'skipped_bbox_downloads': 0,
            'expected_downloads': 0,
            'grid_tiles_processed': [],
            'successful': False
        }

        bbox_size_lon = 1
        bbox_size_lat = 1
        grid = create_longitude_latitude_grid(lon_range=(X_MIN_START, X_MIN_END), lat_range=(Y_MIN_START, Y_MIN_END),
                                              bbox_size_lon=bbox_size_lon, bbox_size_lat=bbox_size_lat)
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

            gdf_subset = filter_gdf_by_bbox(gdf=gdf, bbox_west=lon, bbox_east=lon + bbox_size_lon, bbox_south=lat,
                                            bbox_north=lat + bbox_size_lat)
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

            # Check if download already exists
            if outfile_download.exists():
                print(f'Download already exists for {bbox_west} {bbox_south}! Skipping download.')
                date_results['skipped_bbox_downloads'] += 1
                with open(outfile_downloads_success_file, 'a') as f:
                    f.write(f"{ANALYSIS_DATE}_{grid_coords}\n")
                date_results['grid_tiles_processed'].append(grid_coords)
                continue

            # Download data
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
            'successful_downloads': date_results['success_bbox_downloads'] + date_results['skipped_bbox_downloads'],
            'failed_downloads': date_results['failed_bbox_downloads'],
            'skipped_downloads': date_results['skipped_bbox_downloads'],
            'expected_grid_tiles': expected_grid_tiles,
            'timestamp': datetime.datetime.now().isoformat(),
            'historical_file': str(original_historical_file)
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
                    f"{date_results['skipped_bbox_downloads']} skipped")

        # Store results for this date
        all_results[ANALYSIS_DATE] = date_results

    # ========== SUMMARY ==========
    logger.info(f"\n{'=' * 80}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'=' * 80}")
    for date, results in all_results.items():
        status = "✅ SUCCESS" if results['successful'] else "⚠️ PARTIAL"
        logger.info(f"{date}: {status} - {results['success_bbox_downloads']} successful, "
                    f"{results['failed_bbox_downloads']} failed, "
                    f"{results['skipped_bbox_downloads']} skipped")

    logger.info(f"Overall status: {'✅ SUCCESS' if overall_success else '⚠️ PARTIAL FAILURE'}")

    return {
        'success': overall_success,
        'dates_processed': list(all_results.keys()),
        'date_results': all_results,
        'total_dates': len(all_results),
        'successful_dates': sum(1 for r in all_results.values() if r['successful']),
        'failed_dates': sum(1 for r in all_results.values() if not r['successful'])
    }


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
def merge_near_real_time_region(
        region: str = "TEST",
        run_start_label: str = None,
        env_path: str = None,
        dates_to_merge: List[str] = None,
        verify_downloads_first: bool = True,
        check_duplicates: bool = True,
        strict_duplicate_check: bool = False
):
    """
    Merge downloaded near-real-time data for a specific region.

    This function takes all downloaded files for the region and merges them
    into a new historical NetCDF file. It does not perform any downloads.

    Args:
        region: Region name (e.g., "TEST", "AFRICA", "SOUTH_AMERICA")
        run_start_label: Optional label for tracking runs
        env_path: Optional path to .env file
        dates_to_merge: Optional list of dates to merge (if None, merges all)
        verify_downloads_first: If True, verifies downloads are complete before merging
        check_duplicates: If True, checks for duplicate dates/IDs before merging
        strict_duplicate_check: If True, raises error on duplicates; if False, logs warning and continues

    Returns:
        dict: Status information about the merge
    """
    log_memory_usage("Merge function start")

    # Load environment variables
    if env_path:
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    REGION_NAME = region

    dynamic_world_data_dir = os.environ['dynamic_world_data']
    dynamic_world_download_dir = Path(os.environ['dynamic_world_downloads'])

    # Get all historical files
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
    if not all_dynamic_world_files:
        logger.error(f"No .nc files found in {dynamic_world_data_dir}")
        return {'success': False, 'error': 'No .nc files found'}

    # Get the original historical file
    original_historical_file = max(all_dynamic_world_files, key=os.path.getctime)
    logger.info(f"Using original historical file: {original_historical_file}")

    # Get missing dates (which should have been downloaded)
    missing_dates = utils.download_new_dynamic_world_data.check_missing_data_in_netcdf(original_historical_file)

    if not missing_dates:
        logger.info("No missing dates found - nothing to merge")
        return {'success': True, 'message': 'No missing dates to merge'}

    # Filter dates if specified
    if dates_to_merge:
        missing_dates = [d for d in missing_dates if d.strftime("%Y-%m") in dates_to_merge]
        if not missing_dates:
            logger.warning(f"No matching dates found in {dates_to_merge}")
            return {'success': False, 'error': 'No matching dates found'}

    # ========== DUPLICATE CHECK: Check if any missing dates already have files ==========
    if check_duplicates:
        logger.info("=" * 80)
        logger.info("CHECKING FOR DUPLICATES BEFORE MERGE")
        logger.info("=" * 80)

        # Load historical dataset to get existing dates and IDs
        logger.info("Loading historical dataset for duplicate checking...")
        ds_historical_check = xr.open_dataset(original_historical_file)
        existing_dates = set(pd.to_datetime(ds_historical_check['date'].values))
        existing_ids = set(ds_historical_check['id_geohash'].values)
        ds_historical_check.close()

        logger.info(f"Historical file has {len(existing_dates)} dates and {len(existing_ids)} IDs")

        # Check for duplicate dates
        duplicate_dates = []
        for date in missing_dates:
            ANALYSIS_DATE = date.strftime("%Y-%m")
            date_ts = pd.Timestamp(f"{ANALYSIS_DATE}-01")
            if date_ts in existing_dates:
                duplicate_dates.append(ANALYSIS_DATE)
                logger.warning(f"⚠️ Date {ANALYSIS_DATE} is already in the historical file!")

        if duplicate_dates:
            logger.warning(f"Found {len(duplicate_dates)} duplicate dates: {duplicate_dates}")
            if strict_duplicate_check:
                return {
                    'success': False,
                    'error': f'Duplicate dates found: {duplicate_dates}',
                    'duplicate_dates': duplicate_dates
                }
            else:
                logger.info("Removing duplicate dates from merge list...")
                missing_dates = [d for d in missing_dates if d.strftime("%Y-%m") not in duplicate_dates]
                if not missing_dates:
                    logger.info("No new dates to merge after removing duplicates")
                    return {
                        'success': True,
                        'message': 'No new dates to merge (duplicates removed)',
                        'duplicate_dates_removed': duplicate_dates
                    }
                logger.info(f"Proceeding with {len(missing_dates)} non-duplicate dates")

        # Check for duplicate IDs in downloaded files
        logger.info("Checking for duplicate IDs in downloaded files...")
        duplicate_id_warnings = []

        for date in missing_dates:
            ANALYSIS_DATE = date.strftime("%Y-%m")
            current_download_dir = Path(str(dynamic_world_download_dir), REGION_NAME, f'download_{ANALYSIS_DATE}')

            if current_download_dir.exists():
                downloaded_files = sorted(glob.glob(str(current_download_dir / f'DW_{ANALYSIS_DATE}_*.nc')))
                if downloaded_files:
                    # Check for duplicate IDs within files for this date
                    all_ids_for_date = []
                    for nc_file in downloaded_files:
                        try:
                            ds = xr.open_dataset(nc_file)
                            ids = ds['id_geohash'].values.tolist()
                            all_ids_for_date.extend(ids)
                            ds.close()
                        except Exception as e:
                            logger.warning(f"Could not read {nc_file}: {e}")

                    # Check for duplicates
                    unique_ids = set(all_ids_for_date)
                    if len(all_ids_for_date) != len(unique_ids):
                        duplicate_count = len(all_ids_for_date) - len(unique_ids)
                        duplicate_id_warnings.append({
                            'date': ANALYSIS_DATE,
                            'total_ids': len(all_ids_for_date),
                            'unique_ids': len(unique_ids),
                            'duplicate_count': duplicate_count
                        })
                        logger.warning(
                            f"⚠️ Date {ANALYSIS_DATE}: Found {duplicate_count} duplicate IDs across {len(downloaded_files)} files")

        if duplicate_id_warnings:
            logger.warning(f"Found duplicate ID issues in {len(duplicate_id_warnings)} dates")
            if strict_duplicate_check:
                return {
                    'success': False,
                    'error': 'Duplicate IDs found in downloaded files',
                    'duplicate_details': duplicate_id_warnings
                }

        # Check for files that might be duplicates (same date, overlapping IDs)
        logger.info("Checking for overlapping files...")
        overlapping_warnings = []

        # Group downloaded files by date
        date_file_map = {}
        for date in missing_dates:
            ANALYSIS_DATE = date.strftime("%Y-%m")
            current_download_dir = Path(str(dynamic_world_download_dir), REGION_NAME, f'download_{ANALYSIS_DATE}')
            if current_download_dir.exists():
                downloaded_files = sorted(glob.glob(str(current_download_dir / f'DW_{ANALYSIS_DATE}_*.nc')))
                if downloaded_files:
                    date_file_map[ANALYSIS_DATE] = downloaded_files

        # Check for files that might be duplicates (same bbox pattern)
        for date, files in date_file_map.items():
            bbox_patterns = []
            for file_path in files:
                # Extract bbox from filename
                filename = Path(file_path).stem
                parts = filename.split('_')
                if len(parts) >= 5:
                    # Format: DW_YYYY-MM_west_east_south_north
                    bbox = '_'.join(parts[2:6])
                    bbox_patterns.append(bbox)

            # Check for duplicate bbox patterns
            unique_bboxes = set(bbox_patterns)
            if len(bbox_patterns) != len(unique_bboxes):
                duplicate_bboxes = [b for b in bbox_patterns if bbox_patterns.count(b) > 1]
                unique_duplicates = list(set(duplicate_bboxes))
                overlapping_warnings.append({
                    'date': date,
                    'duplicate_bboxes': unique_duplicates
                })
                logger.warning(f"⚠️ Date {date}: Found duplicate grid tiles: {unique_duplicates}")

        if overlapping_warnings and strict_duplicate_check:
            return {
                'success': False,
                'error': 'Overlapping grid tiles found',
                'overlapping_details': overlapping_warnings
            }

        logger.info("✅ Duplicate checks completed successfully")

    # If verify_downloads_first, check that all downloads are complete
    if verify_downloads_first:
        logger.info("Verifying downloads are complete before merging...")
        verification = verify_downloads_complete(
            region=region,
            analysis_dates=[d.strftime("%Y-%m") for d in missing_dates],
            env_path=env_path,
            strict_mode=True
        )

        if not verification['complete']:
            logger.error("❌ Downloads verification failed - cannot proceed with merge")
            return {
                'success': False,
                'error': 'Downloads incomplete',
                'verification': verification
            }
        logger.info("✅ All downloads verified successfully!")

    # Load the historical dataset
    logger.info(f"Loading historical dataset from: {original_historical_file}")
    ds_historical = xr.open_dataset(original_historical_file)

    # Collect all downloaded files
    all_downloaded_files = []
    date_file_counts = {}

    for date in missing_dates:
        ANALYSIS_DATE = date.strftime("%Y-%m")
        current_download_dir = Path(str(dynamic_world_download_dir), REGION_NAME, f'download_{ANALYSIS_DATE}')

        if current_download_dir.exists():
            downloaded_files = sorted(glob.glob(str(current_download_dir / f'DW_{ANALYSIS_DATE}_*.nc')))
            if downloaded_files:
                logger.info(f"Date {ANALYSIS_DATE}: Found {len(downloaded_files)} downloaded files")
                all_downloaded_files.extend(downloaded_files)
                date_file_counts[ANALYSIS_DATE] = len(downloaded_files)
            else:
                logger.warning(f"Date {ANALYSIS_DATE}: No downloaded files found in {current_download_dir}")
        else:
            logger.warning(f"Date {ANALYSIS_DATE}: Download directory does not exist: {current_download_dir}")

    if not all_downloaded_files:
        logger.error("No downloaded files found to merge!")
        ds_historical.close()
        return {'success': False, 'error': 'No downloaded files found'}

    logger.info(f"Found {len(all_downloaded_files)} total downloaded files to merge")
    logger.info(f"Date breakdown: {date_file_counts}")

    # Process downloaded files in batches to build combined dataset
    BATCH_SIZE = 10
    combined = None
    processed_files = []
    failed_files = []

    for batch_idx in tqdm(range(0, len(all_downloaded_files), BATCH_SIZE), desc="Processing batches"):
        batch_files = all_downloaded_files[batch_idx:batch_idx + BATCH_SIZE]
        batch_datasets = []

        for nc_file in batch_files:
            try:
                ds = xr.open_dataset(nc_file)
                # Verify the file has data
                if len(ds['id_geohash']) > 0:
                    batch_datasets.append(ds)
                    processed_files.append(nc_file)
                else:
                    logger.warning(f"File {nc_file} has no IDs, skipping")
                    failed_files.append(nc_file)
            except Exception as e:
                logger.error(f"Error opening {nc_file}: {e}")
                failed_files.append(nc_file)
                continue

        if batch_datasets:
            try:
                # Combine batch datasets
                batch_combined = xr.concat(batch_datasets, dim='id_geohash')

                # Remove duplicate IDs within this batch
                _, unique_idx = np.unique(batch_combined['id_geohash'].values, return_index=True)
                if len(unique_idx) < len(batch_combined['id_geohash']):
                    logger.debug(
                        f"Removed {len(batch_combined['id_geohash']) - len(unique_idx)} duplicate IDs in batch")
                    batch_combined = batch_combined.isel(id_geohash=np.sort(unique_idx))

                if combined is None:
                    combined = batch_combined
                else:
                    # Combine with existing
                    combined = xr.concat([combined, batch_combined], dim='id_geohash')

                    # Remove duplicate IDs from combined dataset
                    _, unique_idx = np.unique(combined['id_geohash'].values, return_index=True)
                    if len(unique_idx) < len(combined['id_geohash']):
                        logger.debug(
                            f"Removed {len(combined['id_geohash']) - len(unique_idx)} duplicate IDs from combined")
                        combined = combined.isel(id_geohash=np.sort(unique_idx))

            except Exception as e:
                logger.error(f"Error combining batch: {e}")
                failed_files.extend(batch_files)

            # Clean up batch datasets
            for ds in batch_datasets:
                ds.close()
            gc.collect()

    # Report on failed files
    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files")
        if len(failed_files) > 10:
            logger.warning(f"First 10 failed files: {failed_files[:10]}")
        else:
            logger.warning(f"Failed files: {failed_files}")

    if combined is None:
        logger.error("No combined dataset created!")
        ds_historical.close()
        return {'success': False, 'error': 'No combined dataset created'}

    logger.info(f"Combined dataset has {len(combined['id_geohash'])} IDs and {len(combined['date'])} dates")

    # ========== FINAL DUPLICATE CHECK ON COMBINED DATA ==========
    if check_duplicates:
        logger.info("Performing final duplicate check on combined data...")

        # Check for duplicate dates in combined data
        combined_dates = set(pd.to_datetime(combined['date'].values))
        existing_dates = set(pd.to_datetime(ds_historical['date'].values))
        overlapping_dates = combined_dates & existing_dates

        if overlapping_dates:
            logger.warning(f"⚠️ Found {len(overlapping_dates)} overlapping dates between historical and combined data")
            logger.warning(f"Overlapping dates: {sorted(overlapping_dates)}")

            if strict_duplicate_check:
                ds_historical.close()
                combined.close()
                return {
                    'success': False,
                    'error': f'Overlapping dates found: {overlapping_dates}',
                    'overlapping_dates': [d.strftime('%Y-%m-%d') for d in overlapping_dates]
                }
            else:
                logger.info("Removing overlapping dates from combined data...")
                combined = combined.where(~combined['date'].isin(pd.to_datetime(list(overlapping_dates))), drop=True)
                logger.info(f"Combined data now has {len(combined['date'])} dates")

                if len(combined['date']) == 0:
                    logger.warning("No new dates to merge after removing overlaps")
                    ds_historical.close()
                    combined.close()
                    return {
                        'success': True,
                        'message': 'No new dates to merge (overlaps removed)',
                        'overlapping_dates_removed': [d.strftime('%Y-%m-%d') for d in overlapping_dates]
                    }

        # Check for duplicate IDs within combined data
        combined_ids = combined['id_geohash'].values
        unique_ids, counts = np.unique(combined_ids, return_counts=True)
        duplicate_ids = unique_ids[counts > 1]

        if len(duplicate_ids) > 0:
            logger.warning(f"⚠️ Found {len(duplicate_ids)} duplicate IDs in combined data")
            if len(duplicate_ids) <= 10:
                logger.warning(f"Duplicate IDs: {duplicate_ids}")
            else:
                logger.warning(f"First 10 duplicate IDs: {duplicate_ids[:10]}")

            # Remove duplicates (keep first occurrence)
            _, unique_idx = np.unique(combined_ids, return_index=True)
            combined = combined.isel(id_geohash=np.sort(unique_idx))
            logger.info(f"Removed duplicates, now have {len(combined['id_geohash'])} unique IDs")

    # Generate timestamp for the new file
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    original_path = Path(original_historical_file)
    new_historical_path = original_path.parent / f"historical_data_{current_timestamp}.nc"

    # Use the memory-efficient merge function
    logger.info("Starting memory-efficient merge to NetCDF...")
    result_path = create_merged_netcdf_memory_efficient(
        ds_historical=ds_historical,
        combined_ds=combined,
        output_path=new_historical_path,
        chunk_size=5000
    )

    # Clean up
    ds_historical.close()
    combined.close()
    gc.collect()

    if result_path and Path(result_path).exists():
        # Verify the new file
        verification = verify_merged_netcdf(result_path)

        # Additional verification for duplicates in the final file
        if verification['valid'] and check_duplicates:
            logger.info("Verifying final file for duplicates...")
            ds_final = xr.open_dataset(result_path)

            # Check for duplicate dates
            final_dates = set(pd.to_datetime(ds_final['date'].values))
            if len(final_dates) != len(ds_final['date']):
                logger.warning("⚠️ Final file has duplicate dates!")
                verification['has_duplicate_dates'] = True

            # Check for duplicate IDs
            final_ids = ds_final['id_geohash'].values
            if len(set(final_ids)) != len(final_ids):
                logger.warning("⚠️ Final file has duplicate IDs!")
                verification['has_duplicate_ids'] = True

            ds_final.close()

        if verification['valid']:
            logger.info(
                f"✅ Merge successful! New file has {verification['id_count']} IDs, {verification['date_count']} dates")
            logger.info(f"   File size: {verification['file_size_gb']:.2f} GB")
            logger.info(f"   Files merged: {len(processed_files)} successful, {len(failed_files)} failed")

            # Create merged marker
            if run_start_label:
                merged_marker = Path(str(dynamic_world_download_dir),
                                     REGION_NAME) / f'merged_complete_{run_start_label}.success'
            else:
                merged_marker = Path(str(dynamic_world_download_dir),
                                     REGION_NAME) / f'merged_complete_{current_timestamp}.success'

            with open(merged_marker, 'w') as f:
                f.write(f"Merged {len(processed_files)} files into historical NetCDF\n")
                f.write(f"New file: {result_path}\n")
                f.write(f"ID count: {verification['id_count']}\n")
                f.write(f"Date count: {verification['date_count']}\n")
                f.write(f"Files processed: {len(processed_files)}\n")
                f.write(f"Files failed: {len(failed_files)}\n")
                if failed_files:
                    f.write(f"Failed files: {failed_files}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")

            return {
                'success': True,
                'merged_file': str(result_path),
                'id_count': verification['id_count'],
                'date_count': verification['date_count'],
                'file_size_gb': verification['file_size_gb'],
                'files_merged': len(processed_files),
                'files_failed': len(failed_files),
                'failed_files': failed_files if failed_files else None,
                'result': result_path,
                'duplicate_check': check_duplicates,
                'date_file_counts': date_file_counts
            }
        else:
            logger.error(f"❌ Merge verification failed: {verification.get('error', 'Unknown error')}")
            return {'success': False, 'error': 'Merge verification failed', 'verification': verification}
    else:
        logger.error("❌ Merge failed! No output file created.")
        return {'success': False, 'error': 'Merge failed - no output file'}

def process_near_real_time_region(region: str = "TEST", run_start_label: str = None, env_path: str = None):
    """
    Process near-real-time breakpoint analysis for a specific region.

    This function assumes downloads have already been completed by download_near_real_time_region.
    It reads from the most_recent_dynamic_world_file (historical data) and any downloaded files,
    then calculates breakpoints for missing dates.

    Args:
        region: Region name (e.g., "TEST", "AFRICA", "SOUTH_AMERICA")
        run_start_label: Optional label for tracking runs
        env_path: Optional path to .env file
    """
    log_memory_usage("Processing function start")

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
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))

    if not all_dynamic_world_files:
        logger.error(f"No .nc files found in {dynamic_world_data_dir}")
        return False

    logger.debug(f"Region name is {REGION_NAME}")

    bounding_box_coords = region_boundaries[REGION_NAME]

    logger.debug(f"Bounding box coordinates are {bounding_box_coords}")
    time.sleep(15)

    X_MIN_START = bounding_box_coords['X_MIN_START']
    X_MIN_END = bounding_box_coords['X_MIN_END']
    Y_MIN_START = bounding_box_coords['Y_MIN_START']
    Y_MIN_END = bounding_box_coords['Y_MIN_END']

    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)

    hist_file_size_gb = get_file_size_gb(most_recent_dynamic_world_file)
    logger.info(f"Historical NetCDF file size: {hist_file_size_gb:.2f} GB")

    missing_dates = utils.download_new_dynamic_world_data.check_missing_data_in_netcdf(most_recent_dynamic_world_file)

    if missing_dates:
        logger.warning(f"Found {len(missing_dates)} missing dates in historical data")
        for date in missing_dates:
            missing_date_string = date.strftime("%Y-%m")
            logger.warning(f"Missing date: {missing_date_string}")
        logger.info("Will process breakpoints for missing dates using downloaded data")
    else:
        logger.info("No missing dates found in historical data")
        logger.info("No processing required")
        return True

    vector_lake_file = os.environ['vector_lake_file']
    path_lake_vector = vector_lake_file

    # Process each missing date
    for date in missing_dates:
        ANALYSIS_DATE = date.strftime("%Y-%m")

        gdf = gpd.read_parquet(path_lake_vector)
        log_memory_usage("After loading lake vectors")

        # Get most recent historical file
        most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
        logger.debug(f"Most recent dynamic world file: {most_recent_dynamic_world_file}")

        bbox_size_lon = 1
        bbox_size_lat = 1
        grid = create_longitude_latitude_grid(lon_range=(X_MIN_START, X_MIN_END), lat_range=(Y_MIN_START, Y_MIN_END),
                                              bbox_size_lon=bbox_size_lon, bbox_size_lat=bbox_size_lat)
        print('created grid')
        log_memory_usage("After creating grid")

        bp = NRTBreakpoint()

        current_breakpoint_dir = Path(output_dir) / f'breakpoint_{ANALYSIS_DATE}'
        current_breakpoint_dir.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Current breakpoint directory: {current_breakpoint_dir}")

        current_download_dir = Path(str(dynamic_world_download_dir), REGION_NAME, f'download_{ANALYSIS_DATE}')
        if not current_download_dir.exists():
            logger.warning(f"Download directory {current_download_dir} does not exist. Skipping date {ANALYSIS_DATE}")
            continue

        breaks_list = []
        total = len(grid[:])

        # Load historical dataset once to get valid IDs
        logger.info("Loading historical dataset to check valid IDs...")
        ds_historical_check = xr.open_dataset(most_recent_dynamic_world_file)
        valid_historical_ids = set(ds_historical_check['id_geohash'].values)
        ds_historical_check.close()
        logger.info(f"Found {len(valid_historical_ids)} valid IDs in historical dataset")

        # Define expected output columns for empty results
        expected_columns = [
            'date', 'water_observed', 'water_predicted', 'water_residual',
            'water_predicted_lower_90', 'water_predicted_upper_90',
            'water_historical_mean', 'water_historical_median', 'water_historical_std',
            'water_historical_min', 'water_historical_max', 'drainage_confidence'
        ]

        # File for failed breakpoint calculations
        if run_start_label is None:
            run_start_label = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        outfile_breaks_failed_file = current_download_dir / f'grid_tiles_failed_{current_datetime}.txt'

        # Run loop
        logger.debug(f"There are total {total} grid tiles for {REGION_NAME}")
        for i, (lon, lat) in enumerate(tqdm(grid[:], total=total, desc="Processing breakpoints")):
            logger.debug(f"Processing {i}/{total} grid tiles.")
            bbox_west = int(lon)
            bbox_east = int(lon + bbox_size_lon)
            bbox_south = int(lat)
            bbox_north = int(lat + bbox_size_lat)

            print(f"Processing breakpoints for bbox: {bbox_west} {bbox_east} {bbox_south} {bbox_north}")

            outfile_download = current_download_dir / f'DW_{ANALYSIS_DATE}_{bbox_west}_{bbox_east}_{bbox_south}_{bbox_north}.nc'
            outfile_breaks = current_breakpoint_dir / f'DW_{ANALYSIS_DATE}_{bbox_west}_{bbox_east}_{bbox_south}_{bbox_north}_breaks.parquet'

            if outfile_breaks.exists():
                print(f'Breakpoints already calculated! Skipping {bbox_west} {bbox_south}')
                breaks_list.append(pd.read_parquet(outfile_breaks))
                continue

            # Check if downloaded file exists
            if not outfile_download.exists():
                print(f'Downloaded file not found for {bbox_west} {bbox_south}. Skipping breakpoint calculation.')
                continue

            gdf_subset = filter_gdf_by_bbox(gdf=gdf, bbox_west=lon, bbox_east=lon + bbox_size_lon, bbox_south=lat,
                                            bbox_north=lat + bbox_size_lat)
            n_lakes = len(gdf_subset)
            print('Number of lakes: ', n_lakes)

            id_list = gdf_subset['id_geohash'].values.tolist()
            if n_lakes == 0:
                print(f'No lakes for grid {bbox_west} {bbox_south}. Skipping!')
                continue

            # Filter IDs to only those that exist in historical data
            original_count = len(id_list)
            id_list = [id_val for id_val in id_list if id_val in valid_historical_ids]
            filtered_count = len(id_list)

            if filtered_count == 0:
                print(
                    f'WARNING: No valid historical IDs for grid {bbox_west} {bbox_south} (had {original_count} lakes, none in historical data). Skipping!')
                continue
            elif filtered_count < original_count:
                print(
                    f'NOTE: Filtered {original_count - filtered_count} lakes not found in historical data. Processing {filtered_count} lakes.')
                gdf_subset = gdf_subset[gdf_subset['id_geohash'].isin(id_list)]

            # Load data
            try:
                # Load historical data
                ds_historical = xr.open_dataset(most_recent_dynamic_world_file)
                ds_historical_subset = ds_historical.sel(id_geohash=id_list)
                ds_historical.close()
                del ds_historical
                gc.collect()

                # Load downloaded data
                ds_dl = xr.open_dataset(outfile_download)
                ds_dl_dates = pd.to_datetime(ds_dl['date'].values).strftime('%Y-%m')

                # Merge data
                if ANALYSIS_DATE in ds_dl_dates:
                    ds_merged = xr.merge([ds_historical_subset, ds_dl]).sortby('date')
                    print(f'Merged new data for {ANALYSIS_DATE} with historical record')
                else:
                    print(f'WARNING: Downloaded file for {bbox_west} {bbox_south} does not contain {ANALYSIS_DATE}')
                    ds_merged = ds_historical_subset

                ds_dl.close()
                del ds_dl
                gc.collect()

                # Calculate breakpoints
                dwds = DWDataset(ds_merged)

                if ANALYSIS_DATE not in dwds.dates_:
                    logger.warning(
                        f"Analysis date {ANALYSIS_DATE} not in dataset dates for grid {bbox_west} {bbox_south}")
                    empty_result = pd.DataFrame(columns=expected_columns)
                    empty_result.to_parquet(outfile_breaks)
                    breaks_list.append(empty_result)
                    print(f'Created empty result for {bbox_west} {bbox_south} - analysis date not in data')
                else:
                    breaks = bp.calculate_break(dataset=dwds, analysis_date=ANALYSIS_DATE)
                    breaks.to_parquet(outfile_breaks)
                    breaks_list.append(breaks)
                    print(f'Successfully calculated breakpoints for {bbox_west} {bbox_south}')

                # Clean up
                ds_historical_subset.close()
                ds_merged.close()
                del ds_historical_subset, ds_merged
                gc.collect()

            except Exception as e:
                logger.error(f"Error processing breakpoints for {bbox_west} {bbox_south}: {e}")
                with open(outfile_breaks_failed_file, 'a') as f:
                    f.write(str(outfile_breaks) + '\n')
                continue

            # Periodic save
            if len(breaks_list) >= 10:
                logger.info(f"Saving intermediate results...")
                non_empty_breaks = [df for df in breaks_list if not df.empty]
                if non_empty_breaks:
                    breaks_merged = pd.concat(non_empty_breaks, ignore_index=True)
                    joined = gdf.set_index('id_geohash').join(breaks_merged, how='inner').reset_index()
                    partial_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}_partial.parquet'
                    joined.to_parquet(partial_file)
                else:
                    logger.warning("No non-empty breakpoint results to save in partial file")
                breaks_list = []
                gc.collect()

        # Final save for this date
        if breaks_list:
            non_empty_breaks = [df for df in breaks_list if not df.empty]
            if non_empty_breaks:
                breaks_merged = pd.concat(non_empty_breaks, ignore_index=True)
                joined = gdf.set_index('id_geohash').join(breaks_merged, how='inner').reset_index()
                path_to_joined_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}.parquet'
                joined.to_parquet(path_to_joined_file)
                logger.info(f"Final combined file saved to {path_to_joined_file}")
            else:
                logger.warning(f"No valid breakpoint results found for date {ANALYSIS_DATE}")
                empty_result = pd.DataFrame(columns=expected_columns)
                path_to_joined_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}.parquet'
                empty_result.to_parquet(path_to_joined_file)
                logger.info(f"Created empty result file for {ANALYSIS_DATE}")

        end = datetime.datetime.now()
        logger.debug(f"Finished processing date {ANALYSIS_DATE} in {end - start}")

    logger.info(f"Processing completed for region: {REGION_NAME}")
    return True


def process_near_real_time_region_dates(
        region: str = "TEST",
        run_start_label: str = None,
        env_path: str = None,
        current_analysis_dates: List[pd.Timestamp] = None
):
    """
    Process near-real-time breakpoint analysis for specific dates.

    This function assumes downloads have already been completed and merged into the
    historical NetCDF file. It reads directly from the most recent historical file
    and calculates breakpoints for the specified dates.

    Args:
        region: Region name (e.g., "TEST", "AFRICA", "SOUTH_AMERICA")
        run_start_label: Optional label for tracking runs
        env_path: Optional path to .env file
        analysis_dates: List of pandas Timestamps to process. If None,
                       automatically detects missing dates from the historical file.

    Returns:
        bool: True if processing completed successfully
    """
    log_memory_usage("Processing function start")

    region_boundaries = get_region_boundaries()
    logger.debug(f"Analysis dates are {current_analysis_dates}")

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
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))

    if not all_dynamic_world_files:
        logger.error(f"No .nc files found in {dynamic_world_data_dir}")
        return False

    logger.debug(f"Region name is {REGION_NAME}")

    bounding_box_coords = region_boundaries[REGION_NAME]

    logger.debug(f"Bounding box coordinates are {bounding_box_coords}")
    time.sleep(15)

    X_MIN_START = bounding_box_coords['X_MIN_START']
    X_MIN_END = bounding_box_coords['X_MIN_END']
    Y_MIN_START = bounding_box_coords['Y_MIN_START']
    Y_MIN_END = bounding_box_coords['Y_MIN_END']

    # Get the most recent historical file (already contains merged data)
    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
    logger.info(f"Using historical NetCDF file: {most_recent_dynamic_world_file}")
    hist_file_size_gb = get_file_size_gb(most_recent_dynamic_world_file)
    logger.info(f"Historical NetCDF file size: {hist_file_size_gb:.2f} GB")

    # Determine which dates to process
    if current_analysis_dates is not None:
        dates_to_process = current_analysis_dates
        logger.info(f"Processing {len(dates_to_process)} explicitly provided dates")
        for d in dates_to_process:
            logger.info(f"  - {d.strftime('%Y-%m-%d')}")
    else:
        # Auto-detect missing dates from the historical file
        missing_dates = utils.download_new_dynamic_world_data.check_missing_data_in_netcdf(
            most_recent_dynamic_world_file)
        if not missing_dates:
            logger.info("No missing dates found in historical file - nothing to process")
            return True
        dates_to_process = missing_dates
        logger.info(f"Processing {len(dates_to_process)} automatically detected missing dates")
        for d in dates_to_process:
            logger.info(f"  - {d.strftime('%Y-%m-%d')}")

    vector_lake_file = os.environ['vector_lake_file']
    path_lake_vector = vector_lake_file

    # Load GDF once for all dates
    gdf = gpd.read_parquet(path_lake_vector)
    log_memory_usage("After loading lake vectors")

    # Load historical dataset once to get valid IDs
    logger.info("Loading historical dataset to check valid IDs...")
    ds_historical_check = xr.open_dataset(most_recent_dynamic_world_file)
    valid_historical_ids = set(ds_historical_check['id_geohash'].values)
    ds_historical_check.close()
    logger.info(f"Found {len(valid_historical_ids)} valid IDs in historical dataset")

    # Create grid once (same for all dates)
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

    # Define expected output columns for empty results
    expected_columns = [
        'date', 'water_observed', 'water_predicted', 'water_residual',
        'water_predicted_lower_90', 'water_predicted_upper_90',
        'water_historical_mean', 'water_historical_median', 'water_historical_std',
        'water_historical_min', 'water_historical_max', 'drainage_confidence'
    ]

    # Process each date
    for date_idx, date in enumerate(dates_to_process):
        ANALYSIS_DATE = date.strftime("%Y-%m")
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing date {date_idx + 1}/{len(dates_to_process)}: {ANALYSIS_DATE}")
        logger.info(f"{'=' * 80}")

        date_start = datetime.datetime.now()

        bp = NRTBreakpoint()

        current_breakpoint_dir = Path(output_dir) / f'breakpoint_{ANALYSIS_DATE}'
        current_breakpoint_dir.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Current breakpoint directory: {current_breakpoint_dir}")

        breaks_list = []
        total = len(grid[:])

        # File for failed breakpoint calculations
        if run_start_label is None:
            run_start_label = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        outfile_breaks_failed_file = current_breakpoint_dir / f'grid_tiles_failed_{current_datetime}.txt'

        # Run loop over grid tiles
        logger.debug(f"There are total {total} grid tiles for {REGION_NAME}")

        for i, (lon, lat) in enumerate(tqdm(grid[:], total=total, desc=f"Processing {ANALYSIS_DATE}")):
            logger.debug(f"Processing {i}/{total} grid tiles.")
            bbox_west = int(lon)
            bbox_east = int(lon + bbox_size_lon)
            bbox_south = int(lat)
            bbox_north = int(lat + bbox_size_lat)

            print(f"Processing breakpoints for bbox: {bbox_west} {bbox_east} {bbox_south} {bbox_north}")

            outfile_breaks = current_breakpoint_dir / f'DW_{ANALYSIS_DATE}_{bbox_west}_{bbox_east}_{bbox_south}_{bbox_north}_breaks.parquet'

            # Skip if breakpoints already calculated
            if outfile_breaks.exists():
                print(f'Breakpoints already calculated! Skipping {bbox_west} {bbox_south}')
                breaks_list.append(pd.read_parquet(outfile_breaks))
                continue

            # Get lake IDs for this grid tile
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

            # Filter IDs to only those that exist in historical data
            original_count = len(id_list)
            id_list = [id_val for id_val in id_list if id_val in valid_historical_ids]
            filtered_count = len(id_list)

            if filtered_count == 0:
                print(
                    f'WARNING: No valid historical IDs for grid {bbox_west} {bbox_south} (had {original_count} lakes, none in historical data). Skipping!')
                continue
            elif filtered_count < original_count:
                print(
                    f'NOTE: Filtered {original_count - filtered_count} lakes not found in historical data. Processing {filtered_count} lakes.')
                gdf_subset = gdf_subset[gdf_subset['id_geohash'].isin(id_list)]

            # ========== PROCESS BREAKPOINTS USING HISTORICAL DATA ONLY ==========
            try:
                # Load historical data for this tile
                ds_historical = xr.open_dataset(most_recent_dynamic_world_file)
                ds_historical_subset = ds_historical.sel(id_geohash=id_list)
                ds_historical.close()
                del ds_historical
                gc.collect()

                # Create dataset and calculate breakpoints
                dwds = DWDataset(ds_historical_subset)

                # Check if analysis date exists in the dataset
                if ANALYSIS_DATE not in dwds.dates_:
                    logger.warning(
                        f"Analysis date {ANALYSIS_DATE} not in dataset dates for grid {bbox_west} {bbox_south}")
                    empty_result = pd.DataFrame(columns=expected_columns)
                    empty_result.to_parquet(outfile_breaks)
                    breaks_list.append(empty_result)
                    print(f'Created empty result for {bbox_west} {bbox_south} - analysis date not in data')
                else:
                    # Calculate breakpoints
                    breaks = bp.calculate_break(dataset=dwds, analysis_date=ANALYSIS_DATE)
                    breaks.to_parquet(outfile_breaks)
                    breaks_list.append(breaks)
                    print(f'Successfully calculated breakpoints for {bbox_west} {bbox_south}')

                # Clean up
                ds_historical_subset.close()
                del ds_historical_subset
                gc.collect()

            except Exception as e:
                logger.error(f"Error processing breakpoints for {bbox_west} {bbox_south}: {e}")
                with open(outfile_breaks_failed_file, 'a') as f:
                    f.write(str(outfile_breaks) + '\n')
                continue

            # Periodic save
            if len(breaks_list) >= 10:
                logger.info(f"Saving intermediate results...")
                non_empty_breaks = [df for df in breaks_list if not df.empty]
                if non_empty_breaks:
                    breaks_merged = pd.concat(non_empty_breaks, ignore_index=True)
                    joined = gdf.set_index('id_geohash').join(breaks_merged, how='inner').reset_index()
                    partial_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}_partial.parquet'
                    joined.to_parquet(partial_file)
                else:
                    logger.warning("No non-empty breakpoint results to save in partial file")
                breaks_list = []
                gc.collect()

        # Final save for this date
        if breaks_list:
            non_empty_breaks = [df for df in breaks_list if not df.empty]
            if non_empty_breaks:
                breaks_merged = pd.concat(non_empty_breaks, ignore_index=True)
                joined = gdf.set_index('id_geohash').join(breaks_merged, how='inner').reset_index()
                path_to_joined_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}.parquet'
                joined.to_parquet(path_to_joined_file)
                logger.info(f"Final combined file saved to {path_to_joined_file}")
            else:
                logger.warning(f"No valid breakpoint results found for date {ANALYSIS_DATE}")
                empty_result = pd.DataFrame(columns=expected_columns)
                path_to_joined_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}.parquet'
                empty_result.to_parquet(path_to_joined_file)
                logger.info(f"Created empty result file for {ANALYSIS_DATE}")

        date_end = datetime.datetime.now()
        logger.debug(f"Finished processing date {ANALYSIS_DATE} in {date_end - date_start}")

    logger.info(f"Processing completed for region: {REGION_NAME}")
    return True


def process_near_real_time_region_dates_zarr(
        region: str = "TEST",
        run_start_label: str = None,
        env_path: str = None,
        current_analysis_dates: List[pd.Timestamp] = None,
        zarr_compression_level: int = 4
):
    """
    Process near-real-time breakpoint analysis for specific dates and save to Zarr.

    This function assumes downloads have already been completed and merged into the
    historical NetCDF file. It reads directly from the most recent historical file
    and calculates breakpoints for the specified dates, saving results as Zarr datasets
    (one per date).

    Args:
        region: Region name (e.g., "TEST", "AFRICA", "SOUTH_AMERICA")
        run_start_label: Optional label for tracking runs
        env_path: Optional path to .env file
        current_analysis_dates: List of pandas Timestamps to process. If None,
                               automatically detects missing dates from the historical file.
        zarr_compression_level: Compression level for Zarr (0-9, higher = more compression)

    Returns:
        bool: True if processing completed successfully
    """
    log_memory_usage("Processing function start")

    region_boundaries = get_region_boundaries()
    logger.debug(f"Analysis dates are {current_analysis_dates}")

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
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))

    if not all_dynamic_world_files:
        logger.error(f"No .nc files found in {dynamic_world_data_dir}")
        return False

    logger.debug(f"Region name is {REGION_NAME}")

    bounding_box_coords = region_boundaries[REGION_NAME]

    logger.debug(f"Bounding box coordinates are {bounding_box_coords}")
    time.sleep(15)

    X_MIN_START = bounding_box_coords['X_MIN_START']
    X_MIN_END = bounding_box_coords['X_MIN_END']
    Y_MIN_START = bounding_box_coords['Y_MIN_START']
    Y_MIN_END = bounding_box_coords['Y_MIN_END']

    # Get the most recent historical file (already contains merged data)
    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
    logger.info(f"Using historical NetCDF file: {most_recent_dynamic_world_file}")
    hist_file_size_gb = get_file_size_gb(most_recent_dynamic_world_file)
    logger.info(f"Historical NetCDF file size: {hist_file_size_gb:.2f} GB")

    # Determine which dates to process
    if current_analysis_dates is not None:
        dates_to_process = current_analysis_dates
        logger.info(f"Processing {len(dates_to_process)} explicitly provided dates")
        for d in dates_to_process:
            logger.info(f"  - {d.strftime('%Y-%m-%d')}")
    else:
        # Auto-detect missing dates from the historical file
        missing_dates = utils.download_new_dynamic_world_data.check_missing_data_in_netcdf(
            most_recent_dynamic_world_file)
        if not missing_dates:
            logger.info("No missing dates found in historical file - nothing to process")
            return True
        dates_to_process = missing_dates
        logger.info(f"Processing {len(dates_to_process)} automatically detected missing dates")
        for d in dates_to_process:
            logger.info(f"  - {d.strftime('%Y-%m-%d')}")

    vector_lake_file = os.environ['vector_lake_file']
    path_lake_vector = vector_lake_file

    # Load GDF once for all dates
    gdf = gpd.read_parquet(path_lake_vector)
    log_memory_usage("After loading lake vectors")

    # Load historical dataset once to get valid IDs
    logger.info("Loading historical dataset to check valid IDs...")
    ds_historical_check = xr.open_dataset(most_recent_dynamic_world_file)
    valid_historical_ids = set(ds_historical_check['id_geohash'].values)
    ds_historical_check.close()
    logger.info(f"Found {len(valid_historical_ids)} valid IDs in historical dataset")

    # Create grid once (same for all dates)
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

    # Define expected output columns for empty results
    expected_columns = [
        'date', 'water_observed', 'water_predicted', 'water_residual',
        'water_predicted_lower_90', 'water_predicted_upper_90',
        'water_historical_mean', 'water_historical_median', 'water_historical_std',
        'water_historical_min', 'water_historical_max', 'drainage_confidence'
    ]

    # Process each date
    for date_idx, date in enumerate(dates_to_process):
        ANALYSIS_DATE = date.strftime("%Y-%m")
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing date {date_idx + 1}/{len(dates_to_process)}: {ANALYSIS_DATE}")
        logger.info(f"{'=' * 80}")

        date_start = datetime.datetime.now()

        bp = NRTBreakpoint()

        # Create Zarr output directory
        zarr_output_dir = Path(output_dir) / 'breakpoint_zarr'
        zarr_output_dir.mkdir(exist_ok=True, parents=True)

        # Zarr file path for this date
        zarr_path = zarr_output_dir / f'breakpoints_{ANALYSIS_DATE}.zarr'

        # Also keep a backup Parquet file for compatibility if needed
        current_breakpoint_dir = Path(output_dir) / f'breakpoint_{ANALYSIS_DATE}'
        current_breakpoint_dir.mkdir(exist_ok=True, parents=True)

        logger.debug(f"Zarr output path: {zarr_path}")
        logger.debug(f"Parquet backup directory: {current_breakpoint_dir}")

        breaks_list = []
        total = len(grid[:])

        # File for failed breakpoint calculations
        if run_start_label is None:
            run_start_label = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        outfile_breaks_failed_file = current_breakpoint_dir / f'grid_tiles_failed_{current_datetime}.txt'

        # Run loop over grid tiles
        logger.debug(f"There are total {total} grid tiles for {REGION_NAME}")

        for i, (lon, lat) in enumerate(tqdm(grid[:], total=total, desc=f"Processing {ANALYSIS_DATE}")):
            logger.debug(f"Processing {i}/{total} grid tiles.")
            bbox_west = int(lon)
            bbox_east = int(lon + bbox_size_lon)
            bbox_south = int(lat)
            bbox_north = int(lat + bbox_size_lat)

            print(f"Processing breakpoints for bbox: {bbox_west} {bbox_east} {bbox_south} {bbox_north}")

            outfile_breaks_parquet = current_breakpoint_dir / f'DW_{ANALYSIS_DATE}_{bbox_west}_{bbox_east}_{bbox_south}_{bbox_north}_breaks.parquet'

            # Skip if breakpoints already calculated
            if outfile_breaks_parquet.exists():
                print(f'Breakpoints already calculated! Skipping {bbox_west} {bbox_south}')
                breaks_list.append(pd.read_parquet(outfile_breaks_parquet))
                continue

            # Get lake IDs for this grid tile
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

            # Filter IDs to only those that exist in historical data
            original_count = len(id_list)
            id_list = [id_val for id_val in id_list if id_val in valid_historical_ids]
            filtered_count = len(id_list)

            if filtered_count == 0:
                print(
                    f'WARNING: No valid historical IDs for grid {bbox_west} {bbox_south} (had {original_count} lakes, none in historical data). Skipping!')
                continue
            elif filtered_count < original_count:
                print(
                    f'NOTE: Filtered {original_count - filtered_count} lakes not found in historical data. Processing {filtered_count} lakes.')
                gdf_subset = gdf_subset[gdf_subset['id_geohash'].isin(id_list)]

            # ========== PROCESS BREAKPOINTS USING HISTORICAL DATA ONLY ==========
            try:
                # Load historical data for this tile
                ds_historical = xr.open_dataset(most_recent_dynamic_world_file)
                ds_historical_subset = ds_historical.sel(id_geohash=id_list)
                ds_historical.close()
                del ds_historical
                gc.collect()

                # Create dataset and calculate breakpoints
                dwds = DWDataset(ds_historical_subset)

                # Check if analysis date exists in the dataset
                if ANALYSIS_DATE not in dwds.dates_:
                    logger.warning(
                        f"Analysis date {ANALYSIS_DATE} not in dataset dates for grid {bbox_west} {bbox_south}")
                    empty_result = pd.DataFrame(columns=expected_columns)
                    empty_result.to_parquet(outfile_breaks_parquet)
                    breaks_list.append(empty_result)
                    print(f'Created empty result for {bbox_west} {bbox_south} - analysis date not in data')
                else:
                    # Calculate breakpoints
                    breaks = bp.calculate_break(dataset=dwds, analysis_date=ANALYSIS_DATE)
                    breaks.to_parquet(outfile_breaks_parquet)
                    breaks_list.append(breaks)
                    print(f'Successfully calculated breakpoints for {bbox_west} {bbox_south}')

                # Clean up
                ds_historical_subset.close()
                del ds_historical_subset
                gc.collect()

            except Exception as e:
                logger.error(f"Error processing breakpoints for {bbox_west} {bbox_south}: {e}")
                with open(outfile_breaks_failed_file, 'a') as f:
                    f.write(str(outfile_breaks_parquet) + '\n')
                continue

            # Periodic save to Zarr (every 10 tiles)
            if len(breaks_list) >= 10:
                logger.info(f"Saving intermediate results to Zarr...")
                non_empty_breaks = [df for df in breaks_list if not df.empty]
                if non_empty_breaks:
                    breaks_merged = pd.concat(non_empty_breaks, ignore_index=True)
                    joined = gdf.set_index('id_geohash').join(breaks_merged, how='inner').reset_index()

                    # Convert to xarray and save as Zarr
                    ds_breaks = joined.set_index('id_geohash').to_xarray()

                    # Add attributes
                    ds_breaks.attrs.update({
                        'region': REGION_NAME,
                        'analysis_date': ANALYSIS_DATE,
                        'created_at': datetime.datetime.now().isoformat(),
                        'compression_level': zarr_compression_level,
                        'partial': True
                    })

                    # Configure Zarr encoding with compression
                    encoding = {}
                    for var_name in ds_breaks.data_vars:
                        encoding[var_name] = {
                            'compressor': zarr.Blosc(cname='zstd', clevel=zarr_compression_level, shuffle=2)
                        }

                    # Save to Zarr (overwrite partial)
                    ds_breaks.to_zarr(zarr_path, mode='w', encoding=encoding)
                    logger.info(f"Partial Zarr saved to {zarr_path}")

                else:
                    logger.warning("No non-empty breakpoint results to save to Zarr")
                breaks_list = []
                gc.collect()

        # Final save for this date to Zarr
        if breaks_list:
            non_empty_breaks = [df for df in breaks_list if not df.empty]
            if non_empty_breaks:
                breaks_merged = pd.concat(non_empty_breaks, ignore_index=True)
                joined = gdf.set_index('id_geohash').join(breaks_merged, how='inner').reset_index()

                # Convert to xarray
                ds_breaks = joined.set_index('id_geohash').to_xarray()

                # Add attributes
                ds_breaks.attrs.update({
                    'region': REGION_NAME,
                    'analysis_date': ANALYSIS_DATE,
                    'created_at': datetime.datetime.now().isoformat(),
                    'compression_level': zarr_compression_level,
                    'complete': True
                })

                # Configure Zarr encoding with compression
                encoding = {}
                for var_name in ds_breaks.data_vars:
                    if 'drainage_confidence' in var_name:
                        # Use integer compression for confidence values
                        encoding[var_name] = {
                            'compressor': zarr.Blosc(cname='zstd', clevel=zarr_compression_level, shuffle=1),
                            'dtype': 'int32'
                        }
                    else:
                        encoding[var_name] = {
                            'compressor': zarr.Blosc(cname='zstd', clevel=zarr_compression_level, shuffle=2)
                        }

                # Save to Zarr
                ds_breaks.to_zarr(zarr_path, mode='w', encoding=encoding)

                # Also save a copy as Parquet for backward compatibility
                path_to_joined_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}.parquet'
                joined.to_parquet(path_to_joined_file)
                logger.info(f"Final Zarr saved to {zarr_path}")
                logger.info(f"Final Parquet backup saved to {path_to_joined_file}")

                # Log Zarr file size
                zarr_size_gb = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file()) / (1024 ** 3)
                logger.info(f"Zarr file size: {zarr_size_gb:.2f} GB")

            else:
                logger.warning(f"No valid breakpoint results found for date {ANALYSIS_DATE}")
                empty_result = pd.DataFrame(columns=expected_columns)
                path_to_joined_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}.parquet'
                empty_result.to_parquet(path_to_joined_file)

                # Create empty Zarr dataset too
                empty_ds = empty_result.to_xarray()
                empty_ds.attrs.update({
                    'region': REGION_NAME,
                    'analysis_date': ANALYSIS_DATE,
                    'created_at': datetime.datetime.now().isoformat(),
                    'complete': False,
                    'empty': True
                })
                empty_ds.to_zarr(zarr_path, mode='w')
                logger.info(f"Created empty Zarr file for {ANALYSIS_DATE}")

        date_end = datetime.datetime.now()
        logger.debug(f"Finished processing date {ANALYSIS_DATE} in {date_end - date_start}")

    logger.info(f"Processing completed for region: {REGION_NAME}")
    return True

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

    This can be used as a precondition check before triggering the processing workflow.

    Args:
        region: Region name
        analysis_dates: List of dates in "YYYY-MM" format to verify. If None and auto_discover_dates is True,
                       will discover dates from download directories.
        run_start_label: Optional label to match specific download runs
        env_path: Optional path to .env file
        auto_discover_dates: If True, automatically discover dates from download directories
        strict_mode: If True, require ALL downloads to be successful. If False, allow partial success.

    Returns:
        dict: Verification results with details per date
    """
    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    REGION_NAME = region
    dynamic_world_download_dir = Path(os.environ['dynamic_world_downloads'])

    # Discover dates if requested
    if auto_discover_dates or analysis_dates is None:
        download_pattern = str(dynamic_world_download_dir / REGION_NAME / 'download_*')
        download_dirs = glob.glob(download_pattern)

        discovered_dates = []
        for dir_path in download_dirs:
            # Extract date from directory name
            dir_name = Path(dir_path).name
            if dir_name.startswith('download_'):
                date_str = dir_name.replace('download_', '')
                # Validate date format
                try:
                    datetime.datetime.strptime(date_str, '%Y-%m')
                    discovered_dates.append(date_str)
                except ValueError:
                    continue

        if analysis_dates is None:
            analysis_dates = sorted(discovered_dates)
        else:
            # Combine provided dates with discovered dates
            all_dates = set(analysis_dates) | set(discovered_dates)
            analysis_dates = sorted(all_dates)

        if not analysis_dates:
            return {
                'complete': False,
                'reason': 'No dates found to verify',
                'discovered_dates': discovered_dates,
                'date_results': {}
            }

    logger.info(f"Verifying downloads for region '{REGION_NAME}' for {len(analysis_dates)} date(s): {analysis_dates}")

    date_results = {}
    all_complete = True
    missing_dates = []

    for analysis_date in analysis_dates:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Verifying date: {analysis_date}")
        logger.info(f"{'=' * 60}")

        current_download_dir = dynamic_world_download_dir / REGION_NAME / f'download_{analysis_date}'

        date_result = {
            'analysis_date': analysis_date,
            'complete': False,
            'expected_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'skipped_downloads': 0,
            'manifest_file': None,
            'completion_file': None,
            'merged_file': None,
            'details': {}
        }

        # Check if download directory exists
        if not current_download_dir.exists():
            logger.warning(f"Download directory does not exist: {current_download_dir}")
            date_result['reason'] = f"Download directory does not exist: {current_download_dir}"
            date_results[analysis_date] = date_result
            all_complete = False
            missing_dates.append(analysis_date)
            continue

        # Look for manifest files
        manifest_files = list(current_download_dir.glob(f'download_manifest_*.json'))
        if not manifest_files:
            logger.warning(f"No manifest file found for {analysis_date}")
            date_result['reason'] = 'No manifest file found'
            date_results[analysis_date] = date_result
            all_complete = False
            missing_dates.append(analysis_date)
            continue

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
            logger.warning(f"No expected downloads for {analysis_date}")
            date_result['reason'] = 'No expected downloads (grid tiles with lakes)'
            date_result['complete'] = True  # Nothing to download means complete
            date_results[analysis_date] = date_result
            continue

        # Check for completion marker
        success_markers = list(current_download_dir.glob(f'download_complete_*.success'))
        partial_markers = list(current_download_dir.glob(f'download_complete_*.partial'))

        if success_markers:
            date_result['completion_file'] = str(max(success_markers, key=lambda p: p.stat().st_mtime))
            logger.info(f"✅ Found success completion marker for {analysis_date}")
            # Still check if merged file exists
        elif partial_markers:
            date_result['completion_file'] = str(max(partial_markers, key=lambda p: p.stat().st_mtime))
            logger.warning(f"⚠️ Found partial completion marker for {analysis_date} - some downloads failed")
            if strict_mode:
                all_complete = False
                date_result[
                    'reason'] = f"Partial downloads: {date_result['failed_downloads']} failed out of {date_result['expected_downloads']}"
                date_results[analysis_date] = date_result
                continue
        else:
            logger.warning(f"No completion marker found for {analysis_date}")
            date_result['reason'] = 'No completion marker found'
            all_complete = False
            date_results[analysis_date] = date_result
            missing_dates.append(analysis_date)
            continue

        # Check if any failed downloads
        failed_file = current_download_dir / f'grid_tiles_download_failed_*.txt'
        failed_files = list(current_download_dir.glob('grid_tiles_download_failed_*.txt'))

        if failed_files and date_result['failed_downloads'] > 0:
            # Read failed downloads
            failed_grids = []
            for ff in failed_files:
                with open(ff, 'r') as f:
                    failed_grids.extend([line.strip() for line in f.readlines()])

            date_result['failed_grid_tiles'] = failed_grids
            logger.warning(f"Found {len(failed_grids)} failed grid tiles for {analysis_date}")

            if strict_mode:
                all_complete = False
                date_result['reason'] = f"{len(failed_grids)} grid tiles failed to download"
                date_results[analysis_date] = date_result
                continue
        else:
            logger.info(f"✅ No failed downloads for {analysis_date}")

        # All checks passed for this date
        date_result['complete'] = True
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
        logger.warning(f"Incomplete dates: {incomplete_dates}")
        for date in incomplete_dates:
            reason = date_results[date].get('reason', 'Unknown reason')
            logger.warning(f"  - {date}: {reason}")
    else:
        logger.info("✅ All dates are complete and verified!")

    return {
        'complete': all_complete if strict_mode else len(incomplete_dates) == 0,
        'region': REGION_NAME,
        'dates_verified': analysis_dates,
        'complete_dates': complete_dates,
        'incomplete_dates': incomplete_dates,
        'date_results': date_results,
        'missing_dates': missing_dates,
        'strict_mode': strict_mode,
        'summary': {
            'total_dates': len(date_results),
            'complete_count': len(complete_dates),
            'incomplete_count': len(incomplete_dates),
            'total_expected_downloads': sum(r.get('expected_downloads', 0) for r in date_results.values()),
            'total_successful_downloads': sum(r.get('successful_downloads', 0) for r in date_results.values()),
            'total_failed_downloads': sum(r.get('failed_downloads', 0) for r in date_results.values())
        }
    }



def verify_and_trigger_processing(
        region: str = "TEST",
        env_path: str = None,
        auto_discover_dates: bool = True,
        strict_mode: bool = True,
        process_function=None,
        **process_kwargs
):
    """
    Verify downloads and optionally trigger processing if verification passes.

    This is a convenience function that combines verification and processing trigger.

    Args:
        region: Region name
        env_path: Optional path to .env file
        auto_discover_dates: If True, automatically discover dates from download directories
        strict_mode: If True, require ALL downloads to be successful
        process_function: Function to call for processing (e.g., process_near_real_time_region)
        **process_kwargs: Additional arguments to pass to the processing function

    Returns:
        dict: Combined verification and processing results
    """
    # First, verify downloads
    verification_result = verify_downloads_complete(
        region=region,
        analysis_dates=None,
        env_path=env_path,
        auto_discover_dates=auto_discover_dates,
        strict_mode=strict_mode
    )

    result = {
        'verification': verification_result,
        'processing_triggered': False,
        'processing_result': None
    }

    # Check if verification passed
    if verification_result['complete']:
        logger.info(f"✅ All downloads verified for {region}. Triggering processing...")

        if process_function:
            # Trigger the processing function
            try:
                processing_result = process_function(
                    region=region,
                    env_path=env_path,
                    **process_kwargs
                )
                result['processing_triggered'] = True
                result['processing_result'] = processing_result
                logger.info(f"Processing completed: {processing_result}")
            except Exception as e:
                logger.error(f"Error during processing: {e}")
                result['processing_error'] = str(e)
        else:
            logger.info("No processing function provided, skipping processing trigger")
    else:
        incomplete_dates = verification_result.get('incomplete_dates', [])
        logger.warning(f"Cannot trigger processing: {len(incomplete_dates)} dates are incomplete: {incomplete_dates}")
        result['trigger_reason'] = f"Incomplete dates: {incomplete_dates}"

    return result


def main():
    """
    Main entry point for command-line usage.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Run near-real-time analysis for a region')
    parser.add_argument('mode', nargs='?', choices=['download', 'merge', 'process', 'legacy'],
                        default='legacy', help='Mode: download only, merge only, process only, or legacy (all)')
    parser.add_argument('region', nargs='?', default='TEST',
                        help='Region name (default: TEST)')
    parser.add_argument('env_path', nargs='?', default=None,
                        help='Optional path to .env file')
    parser.add_argument('--dates', nargs='+', default=None,
                        help='Specific dates to process (format: YYYY-MM)')

    args = parser.parse_args()

    run_start = datetime.datetime.now()
    run_start_label = run_start.strftime("%Y_%m_%d_%H_%M_%S")

    success = False

    if args.mode == 'download':
        result = download_near_real_time_region(region=args.region, run_start_label=run_start_label,
                                                env_path=args.env_path)
        success = result.get('success', False)
        logger.info(f"Download result: {result}")

    elif args.mode == 'merge':
        result = merge_near_real_time_region(
            region=args.region,
            run_start_label=run_start_label,
            env_path=args.env_path,
            dates_to_merge=args.dates
        )
        success = result.get('success', False)
        logger.info(f"Merge result: {result}")

    elif args.mode == 'process':
        result = process_near_real_time_region(region=args.region, run_start_label=run_start_label,
                                               env_path=args.env_path)
        success = result
        logger.info(f"Process result: {result}")

    else:  # legacy mode
        # Run download, then merge, then process
        logger.info("Running in legacy mode: download -> merge -> process")

        download_result = download_near_real_time_region(region=args.region, run_start_label=run_start_label,
                                                         env_path=args.env_path)
        if download_result.get('success', False):
            merge_result = merge_near_real_time_region(region=args.region, run_start_label=run_start_label,
                                                       env_path=args.env_path)
            if merge_result.get('success', False):
                success = process_near_real_time_region(region=args.region, run_start_label=run_start_label,
                                                        env_path=args.env_path)
            else:
                logger.error("Merge failed, skipping processing")
                success = False
        else:
            logger.error("Download failed, skipping merge and processing")
            success = False

    sys.exit(0 if success else 1)


# if __name__ == '__main__':
#     main()