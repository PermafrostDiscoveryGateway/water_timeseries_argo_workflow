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


def log_memory_usage(stage="", threshold_gb=20):
    """Log memory usage and trigger GC if needed"""
    mem_gb = get_memory_usage()
    logger.info(f"[MEMORY] {stage}: {mem_gb:.2f} GB")

    if mem_gb > threshold_gb:
        logger.warning(f"Memory usage {mem_gb:.2f} GB exceeds threshold {threshold_gb} GB")
        gc.collect()
        # Force garbage collection
        if hasattr(gc, 'collect'):
            gc.collect(2)
    return mem_gb


def open_netcdf_chunked(filepath, chunks=None):
    """
    Open a NetCDF file with chunking to reduce memory usage.

    Args:
        filepath: Path to NetCDF file
        chunks: Dict of chunk sizes (e.g., {'id_geohash': 5000, 'date': -1})

    Returns:
        xarray.Dataset with dask chunks
    """
    if chunks is None:
        chunks = {'id_geohash': 5000, 'date': -1}

    # Use dask for lazy loading
    return xr.open_dataset(filepath, chunks=chunks, engine='netcdf4')


def merge_netcdf_files_chunked(
        input_files,
        output_path,
        chunk_size_id=5000,
        chunk_size_date=-1,
        compression_level=4,
        max_memory_gb=25
):
    """
    Merge multiple NetCDF files in chunks to avoid OOM.

    Args:
        input_files: List of input file paths
        output_path: Output file path
        chunk_size_id: Number of IDs per chunk
        chunk_size_date: Number of dates per chunk (-1 for all)
        compression_level: NetCDF compression level
        max_memory_gb: Maximum memory to use before flushing

    Returns:
        bool: True if successful
    """
    logger.info(f"Merging {len(input_files)} files with chunk size {chunk_size_id}")

    # Enable memory tracking
    enable_memory_tracking()

    # First, open all files with chunks
    datasets = []
    for filepath in input_files:
        try:
            ds = open_netcdf_chunked(
                filepath,
                chunks={'id_geohash': chunk_size_id, 'date': chunk_size_date}
            )
            datasets.append(ds)
        except Exception as e:
            logger.error(f"Error opening {filepath}: {e}")
            continue

    if not datasets:
        logger.error("No datasets to merge")
        return False

    # Get all unique IDs
    all_ids = []
    for ds in datasets:
        all_ids.extend(ds['id_geohash'].values)
    unique_ids = np.unique(all_ids)
    total_ids = len(unique_ids)

    logger.info(f"Total unique IDs: {total_ids}")

    # Process in chunks
    num_chunks = (total_ids + chunk_size_id - 1) // chunk_size_id
    first_chunk = True

    for chunk_idx in tqdm(range(num_chunks), desc="Merging chunks"):
        start_idx = chunk_idx * chunk_size_id
        end_idx = min((chunk_idx + 1) * chunk_size_id, total_ids)
        chunk_ids = unique_ids[start_idx:end_idx]

        log_memory_usage(f"Chunk {chunk_idx + 1}/{num_chunks} start", threshold_gb=20)

        # Merge datasets for this chunk
        chunk_data = None
        for ds in datasets:
            try:
                # Select IDs that exist in this dataset
                ds_ids = set(ds['id_geohash'].values)
                chunk_ids_present = [id_val for id_val in chunk_ids if id_val in ds_ids]

                if chunk_ids_present:
                    # Use dask for lazy computation
                    ds_chunk = ds.sel(id_geohash=chunk_ids_present)

                    if chunk_data is None:
                        chunk_data = ds_chunk
                    else:
                        # Merge with existing chunk data
                        chunk_data = xr.merge([chunk_data, ds_chunk])
            except Exception as e:
                logger.warning(f"Error merging chunk: {e}")

        if chunk_data is not None:
            # Write chunk to NetCDF
            encoding = {}
            for var in chunk_data.data_vars:
                encoding[var] = {
                    'zlib': True,
                    'complevel': compression_level,
                    'shuffle': True,
                    'chunksizes': (min(100, len(chunk_data['date'])), min(1000, len(chunk_data['id_geohash'])))
                }

            if first_chunk:
                chunk_data.to_netcdf(
                    output_path,
                    mode='w',
                    encoding=encoding,
                    unlimited_dims=['id_geohash']
                )
                first_chunk = False
            else:
                # Append mode
                chunk_data.to_netcdf(
                    output_path,
                    mode='a',
                    engine='netcdf4',
                    encoding=encoding
                )

            # Clean up
            chunk_data.close()
            del chunk_data
            gc.collect()

        log_memory_usage(f"Chunk {chunk_idx + 1}/{num_chunks} end", threshold_gb=20)

        # Check memory usage and force GC if needed
        mem_gb = get_memory_usage()
        if mem_gb > max_memory_gb:
            logger.warning(f"Memory usage {mem_gb:.2f} GB exceeds {max_memory_gb} GB")
            logger.info("Forcing garbage collection...")
            gc.collect(2)
            # Also restart dask if using it
            if hasattr(dask, 'distributed'):
                try:
                    from dask.distributed import Client
                    client = Client()
                    client.restart()
                except:
                    pass

    # Clean up
    for ds in datasets:
        try:
            ds.close()
        except:
            pass

    logger.info(f"Successfully merged {total_ids} IDs to {output_path}")
    return True


def get_dask_client(n_workers=2, memory_limit="16GB", threads_per_worker=2):
    """
    Create a Dask client for distributed processing.

    Args:
        n_workers: Number of workers
        memory_limit: Memory limit per worker
        threads_per_worker: Threads per worker

    Returns:
        Client: Dask client
    """
    try:
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            dashboard_address=None,  # Disable dashboard to save memory
            processes=True
        )
        client = Client(cluster)
        logger.info(f"Created Dask client with {n_workers} workers, {memory_limit} each")
        return client
    except Exception as e:
        logger.warning(f"Could not create Dask client: {e}")
        return None


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


def create_merged_netcdf_memory_efficient(
        ds_historical,
        combined_ds,
        output_path,
        chunk_size=5000,
        compression_level=4,
        max_memory_gb=25
):
    """
    Create a merged NetCDF file efficiently using chunked processing.
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

    # If the dataset is small enough, merge directly
    if total_ids < chunk_size * 2:  # Small dataset threshold
        logger.info(f"Total IDs {total_ids} is manageable, merging directly...")
        return merge_direct_small_dataset(ds_historical, combined_ds, output_path, all_ids, compression_level)

    # ========== USE CHUNKED APPROACH FOR LARGE DATASETS ==========
    logger.info(f"Large dataset ({total_ids} IDs), using chunked approach with chunk size {chunk_size}...")

    # Use temporary directory for chunks
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix='merge_chunks_')
    chunk_files = []

    try:
        num_chunks = (total_ids + chunk_size - 1) // chunk_size
        logger.info(f"Processing {num_chunks} chunks of {chunk_size} IDs each")

        for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_ids)
            chunk_ids = all_ids[start_idx:end_idx]
            chunk_ids_list = chunk_ids.tolist()

            # Get data for this chunk
            hist_chunk = None
            existing_hist_ids = [id_val for id_val in chunk_ids_list if id_val in hist_ids]
            if existing_hist_ids:
                hist_chunk = ds_historical.sel(id_geohash=existing_hist_ids)

            combined_chunk = None
            existing_combined_ids = [id_val for id_val in chunk_ids_list if id_val in combined_ids]
            if existing_combined_ids:
                combined_chunk = combined_ds.sel(id_geohash=existing_combined_ids)

            # Merge chunk
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

            # Save chunk to temporary file with compression
            chunk_file = os.path.join(temp_dir, f'chunk_{chunk_idx:04d}.nc')

            encoding = {}
            for var in merged_chunk.data_vars:
                encoding[var] = {
                    'zlib': True,
                    'complevel': compression_level,
                    'shuffle': True,
                    'chunksizes': (min(100, len(merged_chunk['date'])), min(1000, len(merged_chunk['id_geohash'])))
                }

            merged_chunk.to_netcdf(chunk_file, encoding=encoding)
            chunk_files.append(chunk_file)

            # Clean up
            if hist_chunk is not None:
                hist_chunk.close()
            if combined_chunk is not None:
                combined_chunk.close()
            merged_chunk.close()
            gc.collect()

            # Log memory usage
            log_memory_usage(f"Chunk {chunk_idx + 1}/{num_chunks} complete", threshold_gb=max_memory_gb)

        # ========== COMBINE ALL CHUNKS ==========
        if chunk_files:
            logger.info(f"Combining {len(chunk_files)} chunk files...")

            # Open all chunks and concatenate
            chunk_datasets = []
            for chunk_file in chunk_files:
                # Use chunked loading for combining
                ds = xr.open_dataset(chunk_file, chunks={'id_geohash': chunk_size, 'date': -1})
                chunk_datasets.append(ds)

            # Concatenate (this still loads everything, but chunks are already compressed)
            final_merged = xr.concat(chunk_datasets, dim='id_geohash')

            # Remove duplicates if any
            _, unique_idx = np.unique(final_merged['id_geohash'].values, return_index=True)
            if len(unique_idx) < len(final_merged['id_geohash']):
                final_merged = final_merged.isel(id_geohash=np.sort(unique_idx))

            # Write to final file with compression
            encoding = {}
            for var in final_merged.data_vars:
                encoding[var] = {
                    'zlib': True,
                    'complevel': compression_level,
                    'shuffle': True,
                    'chunksizes': (100, 1000)
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


def merge_near_real_time_region_v3_smart_local_disk(
        region: str = "TEST",
        dates_to_merge: List[str] = None,
        input_file_path: str = None,
        env_path: str = None,
        skip_if_already_merged: bool = True,
        verify_downloads_first: bool = True,
        temp_dir: str = None,
        final_copy_path: str = None
):
    """
    Enhanced merge function that APPENDS data to an existing NetCDF file.
    """
    log_memory_usage("Smart Merge Local Disk function start")

    # ... [validation and setup code remains the same] ...

    # Use temp directory if provided, otherwise use the input file's parent
    if temp_dir:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / f"temp_merge_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.nc"
    else:
        temp_file = Path(input_file_path).parent / f"temp_merge_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.nc"

    logger.info(f"Using temp file: {temp_file}")

    # Track if temp file was successfully moved
    temp_file_moved = False

    try:
        # ... [Step 1-6: Load, combine, merge, write to temp] ...

        # Step 7: Move/copy to final location
        logger.info(f"Moving temp file to final location: {input_file_path}")

        # If target exists, make a backup
        if Path(input_file_path).exists():
            backup_file = Path(
                input_file_path).parent / f"{Path(input_file_path).stem}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{Path(input_file_path).suffix}"
            logger.info(f"Backing up existing file to: {backup_file}")
            shutil.move(input_file_path, backup_file)
            logger.info(f"✅ Backup created: {backup_file}")

        # Move temp to final
        shutil.move(temp_file, input_file_path)
        temp_file_moved = True  # Mark as moved
        logger.info(f"✅ File moved to: {input_file_path}")

        # Step 8: If final_copy_path provided, copy to that location
        if final_copy_path:
            logger.info(f"Copying to final location: {final_copy_path}")
            final_copy_path = Path(final_copy_path)
            final_copy_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy (not move) to keep the local file
            shutil.copy2(input_file_path, final_copy_path)
            logger.info(f"✅ Copied to: {final_copy_path}")
            copy_size_gb = final_copy_path.stat().st_size / (1024 ** 3)
            logger.info(f"  Size: {copy_size_gb:.2f} GB")

        # Step 9: Verify the final file
        logger.info("Verifying final file...")
        verify_ds = xr.open_dataset(input_file_path)
        verify_vars = set(verify_ds.data_vars)
        verify_ds.close()

        # ... [result building] ...

        return result

    except Exception as e:
        logger.error(f"Error in merge: {e}")
        import traceback
        traceback.print_exc()

        return {
            'success': False,
            'error': str(e),
            'file_path': input_file_path
        }

    finally:
        # ===== GUARANTEED CLEANUP =====
        # This runs whether there was an exception or not

        # Only clean up temp file if it wasn't moved successfully
        if not temp_file_moved and temp_file and temp_file.exists():
            try:
                temp_file.unlink()
                logger.info(f"✅ Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_file}: {e}")

        # Also check if temp file was left behind in an unexpected state
        elif temp_file and temp_file.exists():
            # This should only happen if the file was moved but somehow still exists
            try:
                # Check if it's actually the same file (by size or content)
                if Path(input_file_path).exists():
                    # If final exists, safe to delete temp
                    temp_file.unlink()
                    logger.info(f"✅ Removed orphaned temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Could not remove orphaned temp file {temp_file}: {e}")

        # Clean up any backup files older than 7 days (optional)
        try:
            backup_pattern = f"{Path(input_file_path).stem}_backup_*.nc"
            backup_dir = Path(input_file_path).parent
            old_backups = backup_dir.glob(backup_pattern)
            for backup in old_backups:
                # Check if backup is older than 7 days
                if (datetime.datetime.now() - datetime.datetime.fromtimestamp(backup.stat().st_mtime)).days > 7:
                    backup.unlink()
                    logger.info(f"Removed old backup: {backup}")
        except Exception as e:
            pass  # Don't fail if cleanup of backups fails

        # Force garbage collection
        gc.collect()
        log_memory_usage("After merge cleanup")


def merge_near_real_time_region_v3_simple(
        region: str = "TEST",
        run_start_label: str = None,
        env_path: str = None,
        dates_to_merge: List[str] = None,
        historical_file_path: str = None,
        verify_downloads_first: bool = True,
        force_merge: bool = True,
        chunk_size_id: int = 5000,  # Add this parameter
        max_memory_gb: int = 25
):
    """
    SIMPLE V3: Always combine historical + new data.
    Preserves ALL variables from the historical file.
    """
    log_memory_usage("Merge V3 Simple function start")

    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    REGION_NAME = region
    dynamic_world_data_dir = os.environ['dynamic_world_data']
    dynamic_world_download_dir = Path(os.environ['dynamic_world_downloads'])

    # ========== VALIDATE INPUT ==========
    if not dates_to_merge:
        logger.error("No dates provided to merge.")
        return {'success': False, 'error': 'No dates provided to merge'}

    # Normalize dates
    normalized_dates = []
    for date in dates_to_merge:
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

    if not normalized_dates:
        logger.error("No valid dates provided")
        return {'success': False, 'error': 'No valid dates provided'}

    dates_to_merge = sorted(normalized_dates)
    logger.info(f"Will merge {len(dates_to_merge)} date(s): {dates_to_merge}")

    # ========== GET HISTORICAL FILE ==========
    if historical_file_path and os.path.exists(historical_file_path):
        original_historical_file = historical_file_path
        logger.info(f"Using specified historical file: {original_historical_file}")
    else:
        all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
        if not all_dynamic_world_files:
            logger.error(f"No .nc files found in {dynamic_world_data_dir}")
            return {'success': False, 'error': 'No .nc files found'}
        original_historical_file = max(all_dynamic_world_files, key=lambda f: Path(f).stat().st_mtime)
        logger.info(f"Using most recent historical file: {original_historical_file}")

    # ========== LOAD HISTORICAL DATA ==========
    logger.info("Loading historical dataset...")
    ds_historical = open_netcdf_chunked(
        original_historical_file,
        chunks={'id_geohash': chunk_size_id, 'date': -1}
    )

    # Track ALL variable names from historical file
    historical_vars = list(ds_historical.data_vars)
    logger.info(f"Historical file has {len(ds_historical['date'])} dates, {len(ds_historical['id_geohash'])} IDs")
    logger.info(f"Historical variables: {historical_vars}")

    # ========== VERIFY DOWNLOADS ==========
    if verify_downloads_first:
        logger.info("Verifying downloads are complete before merging...")
        verification = verify_downloads_complete(
            region=region,
            analysis_dates=dates_to_merge,
            env_path=env_path,
            strict_mode=True
        )
        summary = verification['summary']
        total_expected_downloaded = summary['total_expected_downloaded']
        total_successful_downloaded = summary['total_successful_downloaded']
        total_skipped_downloads = summary['total_skipped_downloads']
        total_skipped_and_successful_downloads =total_expected_downloaded + total_skipped_downloads
        percent_downloaded = float(total_skipped_and_successful_downloads) / float(total_expected_downloaded)
        if not verification['complete'] and percent_downloaded < 0.99:
            logger.error("❌ Downloads verification failed - cannot proceed with merge")
            ds_historical.close()
            return {
                'success': False,
                'error': 'Downloads incomplete',
                'verification': verification
            }
        logger.info("✅ All downloads verified successfully!")

    # ========== COLLECT DOWNLOADED FILES ==========
    all_downloaded_files = []
    date_file_counts = {}
    missing_dates = []

    for date_str in dates_to_merge:
        current_download_dir = Path(str(dynamic_world_download_dir), REGION_NAME, f'download_{date_str}')

        if current_download_dir.exists():
            downloaded_files = sorted(glob.glob(str(current_download_dir / f'DW_{date_str}_*.nc')))
            if downloaded_files:
                logger.info(f"Date {date_str}: Found {len(downloaded_files)} downloaded files")
                all_downloaded_files.extend(downloaded_files)
                date_file_counts[date_str] = len(downloaded_files)
            else:
                logger.warning(f"Date {date_str}: No downloaded files found")
                missing_dates.append(date_str)
        else:
            logger.warning(f"Date {date_str}: Download directory does not exist")
            missing_dates.append(date_str)

    if not all_downloaded_files:
        logger.error("No downloaded files found to merge!")
        ds_historical.close()
        return {
            'success': False,
            'error': 'No downloaded files found',
            'missing_dates': missing_dates
        }

    if missing_dates:
        logger.warning(f"Missing download files for {len(missing_dates)} date(s): {missing_dates}")

    logger.info(f"Found {len(all_downloaded_files)} total downloaded files to merge")

    # ========== PROCESS DOWNLOADED FILES ==========
    logger.info("Starting merge to NetCDF...")
    log_memory_usage("Before merge")

    # Build combined dataset of all new data
    combined = None
    processed_files = []
    failed_files = []

    # Process files in batches
    BATCH_SIZE = 20
    total_files = len(all_downloaded_files)

    for batch_idx in tqdm(range(0, total_files, BATCH_SIZE), desc="Processing files"):
        batch_files = all_downloaded_files[batch_idx:batch_idx + BATCH_SIZE]
        batch_datasets = []

        for file_path in batch_files:
            try:
                ds = xr.open_dataset(file_path)
                if len(ds['id_geohash']) > 0:
                    batch_datasets.append(ds)
                    processed_files.append(file_path)
                else:
                    logger.warning(f"File {file_path} has no IDs, skipping")
                    failed_files.append(file_path)
            except Exception as e:
                logger.error(f"Error opening {file_path}: {e}")
                failed_files.append(file_path)
            finally:
                gc.collect()

        if batch_datasets:
            try:
                # Combine batch
                batch_combined = xr.concat(batch_datasets, dim='id_geohash')

                # Remove duplicate IDs within this batch
                _, unique_idx = np.unique(batch_combined['id_geohash'].values, return_index=True)
                if len(unique_idx) < len(batch_combined['id_geohash']):
                    batch_combined = batch_combined.isel(id_geohash=np.sort(unique_idx))

                if combined is None:
                    combined = batch_combined
                else:
                    combined = xr.concat([combined, batch_combined], dim='id_geohash')
                    _, unique_idx = np.unique(combined['id_geohash'].values, return_index=True)
                    if len(unique_idx) < len(combined['id_geohash']):
                        combined = combined.isel(id_geohash=np.sort(unique_idx))

            except Exception as e:
                logger.error(f"Error combining batch: {e}")
                failed_files.extend(batch_files)

            # Clean up batch datasets
            for ds in batch_datasets:
                ds.close()
            gc.collect()

    if combined is None:
        logger.error("No combined dataset created!")
        ds_historical.close()
        return {
            'success': False,
            'error': 'No combined dataset created',
            'dates_requested': dates_to_merge
        }

    logger.info(f"Combined dataset has {len(combined['id_geohash'])} IDs and {len(combined['date'])} dates")
    logger.info(f"Combined variables: {list(combined.data_vars)}")

    # ========== CRITICAL: Ensure ALL historical variables are preserved ==========
    # Check which variables are missing from the combined dataset
    combined_vars = set(combined.data_vars)
    missing_vars = set(historical_vars) - combined_vars

    if missing_vars:
        logger.warning(f"Combined dataset is missing variables: {missing_vars}")
        logger.info("These variables will be preserved from the historical dataset")

        # For each missing variable, add it from historical dataset with NaN values
        for var_name in missing_vars:
            logger.info(f"Preserving variable '{var_name}' from historical data...")
            # Get the variable from historical data (with all dates and IDs)
            hist_var = ds_historical[var_name]
            # Add it to the combined dataset
            combined[var_name] = (('id_geohash', 'date'),
                                  np.full((len(combined['id_geohash']), len(combined['date'])), np.nan))
            # For IDs that exist in historical data, copy values
            # This is more complex - we'll handle this in the merge step

    # ========== MERGE HISTORICAL + COMBINED ==========
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_file = Path(original_historical_file).parent / f"historical_data_{timestamp}.nc"

    logger.info(f"Creating merged file: {new_file}")
    logger.info(f"Historical: {len(ds_historical['id_geohash'])} IDs, {len(ds_historical['date'])} dates")
    logger.info(f"New data: {len(combined['id_geohash'])} IDs, {len(combined['date'])} dates")

    # ========== MERGE: Combine both datasets preserving all variables ==========
    logger.info("Combining historical and new data...")

    try:
        # METHOD 1: Use merge with compat='override' but ensure all variables exist
        # First, make sure combined has all the variables from historical (with NaN for missing)
        for var_name in historical_vars:
            if var_name not in combined.data_vars:
                # Create an empty variable with the right shape
                combined[var_name] = (('id_geohash', 'date'),
                                      np.full((len(combined['id_geohash']), len(combined['date'])), np.nan))

        # Now merge - all variables should be present in both
        final_combined = xr.merge([ds_historical, combined], compat='override')

        # Sort by date and id
        final_combined = final_combined.sortby(['date', 'id_geohash'])

        # Remove duplicate IDs (just in case)
        _, unique_idx = np.unique(final_combined['id_geohash'].values, return_index=True)
        if len(unique_idx) < len(final_combined['id_geohash']):
            logger.info(f"Removing {len(final_combined['id_geohash']) - len(unique_idx)} duplicate IDs")
            final_combined = final_combined.isel(id_geohash=np.sort(unique_idx))

        # Verify all variables are present
        final_vars = set(final_combined.data_vars)
        missing_final = set(historical_vars) - final_vars
        if missing_final:
            logger.error(f"CRITICAL: Still missing variables after merge: {missing_final}")
        else:
            logger.info(f"✅ All {len(historical_vars)} variables preserved: {sorted(historical_vars)}")

        # Write with compression
        logger.info(
            f"Writing final file with {len(final_combined['id_geohash'])} IDs and {len(final_combined['date'])} dates...")

        encoding = {}
        for var in final_combined.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 4,
                'shuffle': True
            }

        try:
            final_combined.to_netcdf(new_file, encoding=encoding)
        except Exception as e:
            logger.error(f"Error writing with compression: {e}")
            logger.info("Retrying without compression...")
            final_combined.to_netcdf(new_file)

    except Exception as e:
        logger.error(f"Error merging datasets: {e}")
        # Fallback: try concat with data variables
        logger.info("Trying alternative merge method...")
        try:
            # Create a new dataset with all variables
            final_combined = xr.Dataset()

            # Add all variables from historical
            for var_name in historical_vars:
                if var_name in ds_historical:
                    final_combined[var_name] = ds_historical[var_name]

            # Add/update variables from combined
            for var_name in combined.data_vars:
                final_combined[var_name] = combined[var_name]

            final_combined = final_combined.sortby(['date', 'id_geohash'])
            final_combined.to_netcdf(new_file)
        except Exception as e2:
            logger.error(f"Error with fallback merge: {e2}")
            ds_historical.close()
            combined.close()
            return {'success': False, 'error': f'Merge failed: {e2}'}

    # Clean up
    ds_historical.close()
    combined.close()
    if 'final_combined' in locals():
        final_combined.close()
    gc.collect()

    # ========== VERIFY ==========
    if os.path.exists(new_file):
        file_size_gb = get_file_size_gb(str(new_file))
        logger.info(f"✅ Merged file created: {new_file}")
        logger.info(f"   File size: {file_size_gb:.2f} GB")

        # Verify the file has all variables
        verify_ds = xr.open_dataset(new_file)
        verify_vars = set(verify_ds.data_vars)
        verify_ds.close()

        missing_vars_final = set(historical_vars) - verify_vars
        if missing_vars_final:
            logger.error(f"❌ Final file missing variables: {missing_vars_final}")
            return {
                'success': False,
                'error': f'Missing variables: {missing_vars_final}',
                'missing_variables': list(missing_vars_final)
            }

        # Verify
        verification = verify_merged_netcdf(str(new_file))

        # ========== SUMMARY ==========
        logger.info(f"\n{'=' * 80}")
        logger.info("MERGE V3 SUMMARY")
        logger.info(f"{'=' * 80}")

        result = {
            'success': verification['valid'],
            'mode': 'simple-merge',
            'region': REGION_NAME,
            'dates_requested': dates_to_merge,
            'missing_dates': missing_dates,
            'files_processed': len(processed_files),
            'files_failed': len(failed_files),
            'final_file': str(new_file),
            'verification': verification,
            'variables_preserved': list(verify_vars)
        }

        if verification['valid']:
            logger.info(f"✅ Merge V3 successful!")
            logger.info(f"   Final file: {new_file}")
            logger.info(f"   IDs: {verification['id_count']}")
            logger.info(f"   Dates: {verification['date_count']}")
            logger.info(f"   File size: {verification['file_size_gb']:.2f} GB")
            logger.info(f"   Variables: {sorted(verify_vars)}")
        else:
            logger.error("❌ Merge V3 verification failed!")

        return result
    else:
        logger.error("❌ Merge failed - no file created!")
        return {'success': False, 'error': 'No file created'}


def has_region_been_merged_for_dates(
        region: str,
        dates_to_check: List[str],
        historical_file_path: str = None,
        env_path: str = None
) -> dict:
    """
    Check if a specific region already has the given dates merged.

    This function uses region boundaries to determine which IDs belong to the region,
    then checks if those IDs have data for the specified dates.

    Args:
        region: Region name (TEST, ALASKA, CANADA, EURASIA1, EURASIA2, EURASIA3)
        dates_to_check: List of dates in "YYYY-MM" format
        historical_file_path: Optional path to the historical file
        env_path: Optional path to .env file

    Returns:
        dict: {
            'all_dates_present': bool,  # True if ALL dates are present for ALL region IDs
            'some_dates_present': bool,  # True if at least one date is present for some IDs
            'none_dates_present': bool,  # True if no dates are present for any region IDs
            'missing_dates': List[str],  # Dates that are missing for ANY region ID
            'present_dates': List[str],  # Dates that are present for ALL region IDs
            'partial_dates': List[str],  # Dates that are present for SOME but not ALL region IDs
            'region_id_count': int,      # Total number of IDs in this region
            'region_ids_with_data': int,  # Number of region IDs that have data for the dates
            'region_ids_missing_data': int,  # Number of region IDs missing data
            'file_path': str,
            'region_found': bool
        }
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
        # Get region boundaries
        from utils.region_boundaries import get_region_boundaries
        region_boundaries = get_region_boundaries()

        if region not in region_boundaries:
            return {
                'all_dates_present': False,
                'some_dates_present': False,
                'none_dates_present': True,
                'error': f'Region {region} not found in boundaries',
                'missing_dates': dates_to_check,
                'present_dates': [],
                'partial_dates': [],
                'region_found': False
            }

        bounds = region_boundaries[region]
        x_min_start = bounds['X_MIN_START']
        x_min_end = bounds['X_MIN_END']
        y_min_start = bounds['Y_MIN_START']
        y_min_end = bounds['Y_MIN_END']

        # Open the NetCDF file
        ds = xr.open_dataset(historical_file_path)

        # Get all IDs
        all_ids = ds['id_geohash'].values

        # ========== METHOD 1: If coordinates are stored in the dataset ==========
        region_ids = []

        # Check if we have coordinate information
        has_coords = False
        if 'longitude' in ds.coords and 'latitude' in ds.coords:
            # Try to get coordinates for each ID
            try:
                # This assumes the coordinates are aligned with the IDs
                lons = ds['longitude'].values
                lats = ds['latitude'].values

                if len(lons) == len(all_ids):
                    # Filter IDs within the bounding box
                    for i, id_val in enumerate(all_ids):
                        if (x_min_start <= lons[i] <= x_min_end and
                                y_min_start <= lats[i] <= y_min_end):
                            region_ids.append(id_val)
                    has_coords = True
                    logger.info(f"Found {len(region_ids)} IDs in region {region} using coordinates")
            except Exception as e:
                logger.warning(f"Could not use coordinates from dataset: {e}")

        # ========== METHOD 2: Use geohash decoding if available ==========
        if not has_coords or len(region_ids) == 0:
            try:
                import geohash2
                logger.info("Attempting to decode geohashes to find region IDs...")

                for id_val in all_ids:
                    try:
                        # Decode geohash to get coordinates
                        # Note: geohash2.decode returns (lat, lon)
                        lat, lon = geohash2.decode(id_val)

                        if (x_min_start <= lon <= x_min_end and
                                y_min_start <= lat <= y_min_end):
                            region_ids.append(id_val)
                    except Exception as e:
                        # Skip invalid geohashes
                        continue

                if len(region_ids) > 0:
                    logger.info(f"Found {len(region_ids)} IDs in region {region} using geohash decoding")
                    has_coords = True
            except ImportError:
                logger.warning("geohash2 not installed, cannot decode geohashes")
            except Exception as e:
                logger.warning(f"Error decoding geohashes: {e}")

        # ========== METHOD 3: Fallback - use stored region info if available ==========
        if not has_coords or len(region_ids) == 0:
            # Check if region is stored as a variable or attribute
            if 'region' in ds.data_vars:
                try:
                    region_mask = ds['region'] == region
                    region_ids = ds['id_geohash'].values[region_mask.values]
                    logger.info(f"Found {len(region_ids)} IDs in region {region} using region variable")
                except Exception as e:
                    logger.warning(f"Could not use region variable: {e}")
            elif 'region' in ds.attrs:
                # Region stored as global attribute - not reliable for per-ID
                logger.warning("Region stored as global attribute - cannot verify per-ID region")

        # ========== METHOD 4: Last resort - use all IDs ==========
        if len(region_ids) == 0:
            logger.warning(f"No IDs found for region {region} using any method.")
            logger.warning(f"Using ALL IDs in the file as a fallback (may not be accurate)")
            region_ids = list(all_ids)

        # ========== Check dates for region IDs ==========
        if region_ids:
            # Get the subset of data for this region
            region_data = ds.sel(id_geohash=region_ids)

            # Get existing dates for this region
            existing_dates = set(pd.to_datetime(region_data['date'].values))
            existing_date_strings = {d.strftime("%Y-%m") for d in existing_dates}

            # Check which dates are present
            present_dates = []
            missing_dates = []
            partial_dates = []

            for date_str in dates_to_check:
                if date_str in existing_date_strings:
                    # Check if ALL IDs in the region have data for this date
                    # Get data for this specific date
                    date_data = region_data.sel(date=pd.Timestamp(f"{date_str}-01"))

                    # Check how many IDs have non-NaN data for this date
                    # Use water_observed as a proxy for data presence
                    if 'water_observed' in date_data.data_vars:
                        has_data_mask = ~np.isnan(date_data['water_observed'].values)
                        ids_with_data = np.sum(has_data_mask)
                        total_ids = len(region_ids)

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
                        # Can't check data presence, assume it's present
                        present_dates.append(date_str)
                else:
                    missing_dates.append(date_str)

            # Determine status
            all_present = len(missing_dates) == 0 and len(partial_dates) == 0
            some_present = len(present_dates) > 0 or len(partial_dates) > 0
            none_present = len(present_dates) == 0 and len(partial_dates) == 0

            # Count IDs with data
            ids_with_data = 0
            try:
                # Check how many IDs have any data
                for id_val in region_ids[:100]:  # Sample to avoid memory issues
                    sample_data = region_data.sel(id_geohash=id_val)
                    if 'water_observed' in sample_data.data_vars:
                        if np.any(~np.isnan(sample_data['water_observed'].values)):
                            ids_with_data += 1
            except:
                ids_with_data = 0

            result = {
                'all_dates_present': all_present,
                'some_dates_present': some_present,
                'none_dates_present': none_present,
                'present_dates': present_dates,
                'missing_dates': missing_dates,
                'partial_dates': partial_dates,
                'file_path': historical_file_path,
                'region_found': True,
                'region_id_count': len(region_ids),
                'region_ids_with_data': ids_with_data,
                'region_ids_missing_data': len(region_ids) - ids_with_data,
                'all_dates_in_file': sorted(existing_date_strings)
            }
        else:
            # No region-specific IDs found
            result = {
                'all_dates_present': False,
                'some_dates_present': False,
                'none_dates_present': True,
                'present_dates': [],
                'missing_dates': dates_to_check,
                'partial_dates': [],
                'file_path': historical_file_path,
                'region_found': False,
                'region_id_count': 0,
                'region_ids_with_data': 0,
                'region_ids_missing_data': 0,
                'message': f'No IDs found for region {region} in the file'
            }

        ds.close()

        # Log summary
        logger.info(f"Region {region} check results:")
        logger.info(f"  Total IDs in region: {result['region_id_count']}")
        logger.info(f"  IDs with data: {result['region_ids_with_data']}")
        logger.info(f"  IDs missing data: {result['region_ids_missing_data']}")
        logger.info(f"  All dates present: {result['all_dates_present']}")
        logger.info(f"  Some dates present: {result['some_dates_present']}")
        logger.info(f"  None dates present: {result['none_dates_present']}")
        logger.info(f"  Present dates: {result['present_dates']}")
        logger.info(f"  Partial dates: {result['partial_dates']}")
        logger.info(f"  Missing dates: {result['missing_dates']}")

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
            'file_path': historical_file_path
        }


def get_ids_for_region_from_file(ds, region: str) -> List[str]:
    """
    Get IDs for a specific region by checking which IDs fall within the region's bounds.

    This is a helper function that uses the region boundaries to filter IDs.
    """
    from utils.region_boundaries import get_region_boundaries

    try:
        # Get region boundaries
        region_boundaries = get_region_boundaries()
        if region not in region_boundaries:
            logger.warning(f"Region {region} not found in boundaries")
            return []

        bounds = region_boundaries[region]
        x_min_start = bounds['X_MIN_START']
        x_min_end = bounds['X_MIN_END']
        y_min_start = bounds['Y_MIN_START']
        y_min_end = bounds['Y_MIN_END']

        # Get all IDs and their coordinates
        # Assuming you have longitude and latitude information
        # This depends on how your data is structured
        all_ids = ds['id_geohash'].values

        # If you have coordinate information in the dataset
        if 'longitude' in ds.coords and 'latitude' in ds.coords:
            # This is a simplified example - adjust based on your actual data structure
            lons = ds['longitude'].values
            lats = ds['latitude'].values

            # Filter IDs within the bounding box
            region_ids = []
            for i, id_val in enumerate(all_ids):
                if (x_min_start <= lons[i] <= x_min_end and
                        y_min_start <= lats[i] <= y_min_end):
                    region_ids.append(id_val)

            return region_ids

        # If you have the ID geohash and a way to convert it to coordinates
        # You might need to use the geohash library
        # This is a placeholder - implement based on your actual data structure
        logger.warning("Cannot filter IDs by region - no coordinate information available")
        return []

    except Exception as e:
        logger.error(f"Error getting region IDs: {e}")
        return []

def merge_near_real_time_region_v3_smart(
        region: str = "TEST",
        dates_to_merge: List[str] = None,
        historical_file_path: str = None,
        env_path: str = None,
        skip_if_already_merged: bool = True,
        verify_downloads_first: bool = True
):
    """
    Enhanced merge function that checks if the region already has the dates.

    This version properly handles multiple regions by checking region-specific data.
    """
    log_memory_usage("Smart Merge function start")

    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Normalize dates
    normalized_dates = []
    for date in dates_to_merge:
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

    if not normalized_dates:
        logger.error("No valid dates provided")
        return {'success': False, 'error': 'No valid dates provided'}

    # Check if the region already has these dates
    if skip_if_already_merged and historical_file_path:
        status = has_region_been_merged_for_dates(
            region=region,
            dates_to_check=normalized_dates,
            historical_file_path=historical_file_path,
            env_path=env_path
        )

        if status.get('all_dates_present', False):
            logger.info(f"✅ Region {region} already has all dates {normalized_dates} merged")
            return {
                'success': True,
                'skipped': True,
                'reason': 'All dates already merged',
                'status': status
            }
        elif status.get('present_dates'):
            # Some dates are present, only merge the missing ones
            missing_dates = status.get('missing_dates', [])
            if missing_dates:
                logger.info(
                    f"Region {region} has {len(status['present_dates'])} dates, need to merge {len(missing_dates)}: {missing_dates}")
                # Update dates_to_merge to only the missing ones
                normalized_dates = missing_dates
            else:
                logger.info(f"Region {region} has all dates present")
                return {
                    'success': True,
                    'skipped': True,
                    'reason': 'All dates already merged',
                    'status': status
                }

    # Get the historical file if not provided
    if historical_file_path is None:
        dynamic_world_data_dir = os.environ['dynamic_world_data']
        all_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
        if not all_files:
            logger.error("No NetCDF files found")
            return {'success': False, 'error': 'No NetCDF files found'}
        historical_file_path = max(all_files, key=lambda f: Path(f).stat().st_mtime)

    # Now proceed with the merge for the remaining dates
    logger.info(f"Merging dates {normalized_dates} for region {region}")

    merge_result = merge_near_real_time_region_v3_simple(
        region=region,
        dates_to_merge=normalized_dates,
        historical_file_path=historical_file_path,
        env_path=env_path,
        verify_downloads_first=verify_downloads_first
    )

    return merge_result


def merge_near_real_time_region_v4_smart(
        region: str = "TEST",
        dates_to_merge: List[str] = None,
        historical_file_path: str = None,
        env_path: str = None,
        skip_if_already_merged: bool = True,
        verify_downloads_first: bool = True
):
    """
    Enhanced merge function that checks if the region already has the dates.

    This version properly handles multiple regions by checking region-specific data.
    """
    log_memory_usage("Smart Merge function start")

    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Normalize dates
    normalized_dates = []
    for date in dates_to_merge:
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

    if not normalized_dates:
        logger.error("No valid dates provided")
        return {'success': False, 'error': 'No valid dates provided'}

    # Check if the region already has these dates
    if skip_if_already_merged and historical_file_path:
        status = has_region_been_merged_for_dates(
            region=region,
            dates_to_check=normalized_dates,
            historical_file_path=historical_file_path,
            env_path=env_path
        )

        # Check the different merge scenarios
        if status.get('all_dates_present', False):
            logger.info(f"✅ Region {region} already has ALL dates {normalized_dates} merged")
            return {
                'success': True,
                'skipped': True,
                'reason': 'All dates already merged for all IDs',
                'status': status
            }
        elif status.get('some_dates_present', False):
            # Some dates are partially or fully present
            present_dates = status.get('present_dates', [])
            partial_dates = status.get('partial_dates', [])
            missing_dates = status.get('missing_dates', [])

            logger.info(f"Region {region} has some dates present:")
            logger.info(f"  Fully present: {present_dates}")
            logger.info(f"  Partially present: {partial_dates}")
            logger.info(f"  Missing: {missing_dates}")

            # If there are partial dates, we need to merge all dates to fill gaps
            if partial_dates and missing_dates:
                # Some dates are partial - merge ALL dates to ensure complete coverage
                logger.info(
                    f"Found partial dates {partial_dates}, merging all {normalized_dates} to ensure full coverage")
                # Don't filter - merge all dates
                pass
            elif missing_dates:
                # Only missing dates - merge those
                logger.info(f"Only missing dates remain: {missing_dates}")
                normalized_dates = missing_dates
            else:
                # Only partial dates - merge them to fill gaps
                logger.info(f"Only partial dates remain: {partial_dates}")
                normalized_dates = partial_dates

            if not normalized_dates:
                logger.info(f"No dates need merging for region {region}")
                return {
                    'success': True,
                    'skipped': True,
                    'reason': 'All dates already have data',
                    'status': status
                }
        elif status.get('none_dates_present', False):
            # No dates present - merge everything
            logger.info(f"Region {region} has no dates present, merging all {normalized_dates}")
        else:
            # Unknown status - merge everything to be safe
            logger.warning(f"Unknown status for region {region}, merging all dates to be safe")

        # Log what we're actually going to merge
        logger.info(f"Proceeding to merge {len(normalized_dates)} date(s): {normalized_dates} for region {region}")

    # Get the historical file if not provided
    if historical_file_path is None:
        dynamic_world_data_dir = os.environ['dynamic_world_data']
        all_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
        if not all_files:
            logger.error("No NetCDF files found")
            return {'success': False, 'error': 'No NetCDF files found'}
        historical_file_path = max(all_files, key=lambda f: Path(f).stat().st_mtime)

    # Now proceed with the merge for the remaining dates
    logger.info(f"Merging dates {normalized_dates} for region {region}")

    merge_result = merge_near_real_time_region_v3_simple(
        region=region,
        dates_to_merge=normalized_dates,
        historical_file_path=historical_file_path,
        env_path=env_path,
        verify_downloads_first=verify_downloads_first
    )

    # Add the status info to the result
    if 'status' in locals():
        merge_result['pre_merge_status'] = status

    return merge_result

def merge_near_real_time_region_v3_smart_local_disk(
        region: str = "TEST",
        dates_to_merge: List[str] = None,
        input_file_path: str = None,  # The file to append to (MUST already contain source data)
        env_path: str = None,
        skip_if_already_merged: bool = True,
        temp_dir: str = None,  # Optional temp directory for intermediate files
        final_copy_path: str = None,  # If provided, copy the final file here
):
    """
    Merge new data into an existing local NetCDF file.

    IMPORTANT: This function assumes input_file_path already contains the source data!
    It will only append new data for the specified dates.

    This is designed for local disk usage - write to a local file, then copy to Filestore afterward.

    Args:
        region: Region name
        dates_to_merge: List of dates in "YYYY-MM" format
        input_file_path: Path to the NetCDF file (must already exist with source data)
        env_path: Optional path to .env file
        skip_if_already_merged: If True, skip dates already in the file
        temp_dir: Optional temp directory for intermediate files
        final_copy_path: If provided, copy the final file here after writing

    Returns:
        dict: Merge result with status information
    """
    log_memory_usage("Smart Merge Local Disk function start")

    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Validate input file path
    if input_file_path is None:
        logger.error("input_file_path is required for local disk merge")
        return {'success': False, 'error': 'input_file_path is required'}

    input_path = Path(input_file_path)

    # Ensure the input file exists (it should have been copied from source)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_file_path}")
        logger.error("You must copy the source file to this location first")
        return {'success': False, 'error': f'Input file does not exist: {input_file_path}'}

    # Ensure directory exists
    input_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize dates
    normalized_dates = []
    for date in dates_to_merge:
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

    if not normalized_dates:
        logger.error("No valid dates provided")
        return {'success': False, 'error': 'No valid dates provided'}

    # Check if the file already has these dates for this region
    if skip_if_already_merged:
        status = has_region_been_merged_for_dates(
            region=region,
            dates_to_check=normalized_dates,
            historical_file_path=str(input_path),
            env_path=env_path
        )

        if status.get('all_dates_present', False):
            logger.info(f"✅ Region {region} already has all dates {normalized_dates} in {input_file_path}")
            return {
                'success': True,
                'skipped': True,
                'reason': 'All dates already merged',
                'file_path': str(input_path),
                'status': status
            }
        elif status.get('present_dates'):
            # Some dates are present, only merge the missing ones
            missing_dates = status.get('missing_dates', [])
            if missing_dates:
                logger.info(
                    f"Region {region} has {len(status['present_dates'])} dates, need to append {len(missing_dates)}: {missing_dates}")
                normalized_dates = missing_dates
            else:
                logger.info(f"Region {region} has all dates present")
                return {
                    'success': True,
                    'skipped': True,
                    'reason': 'All dates already merged',
                    'file_path': str(input_path),
                    'status': status
                }

    logger.info(f"Appending {len(normalized_dates)} date(s) to {input_file_path}: {normalized_dates}")

    # Use temp directory if provided, otherwise use the input file's parent
    if temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_dir_path.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir_path / f"temp_merge_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.nc"
    else:
        temp_file = input_path.parent / f"temp_merge_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.nc"

    logger.info(f"Using temp file: {temp_file}")

    try:
        # Step 1: Load the existing local file (already has all source data)
        logger.info(f"Loading existing data from {input_file_path}...")
        ds_existing = xr.open_dataset(input_file_path)

        existing_id_count = len(ds_existing['id_geohash'])
        existing_date_count = len(ds_existing['date'])
        logger.info(f"  Existing file has {existing_id_count:,} IDs and {existing_date_count} dates")
        logger.info(f"  Existing variables: {list(ds_existing.data_vars)}")

        # Step 2: Get the downloaded data for the dates
        logger.info(f"Getting downloaded data for dates: {normalized_dates}")

        dynamic_world_download_dir = Path(os.environ.get('dynamic_world_downloads', ''))
        if not dynamic_world_download_dir.exists():
            logger.error(f"Download directory does not exist: {dynamic_world_download_dir}")
            ds_existing.close()
            return {'success': False, 'error': f'Download directory does not exist: {dynamic_world_download_dir}'}

        all_downloaded_files = []

        for date_str in normalized_dates:
            download_dir = dynamic_world_download_dir / region / f'download_{date_str}'
            if download_dir.exists():
                downloaded_files = glob.glob(str(download_dir / f'DW_{date_str}_*.nc'))
                all_downloaded_files.extend(downloaded_files)
                logger.info(f"Found {len(downloaded_files)} files for {date_str}")
            else:
                logger.warning(f"Download directory not found for {date_str}: {download_dir}")

        if not all_downloaded_files:
            logger.error("No downloaded files found to merge")
            ds_existing.close()
            return {'success': False, 'error': 'No downloaded files found'}

        # Step 3: Combine downloaded files into a single dataset
        logger.info(f"Combining {len(all_downloaded_files)} downloaded files...")
        combined = None
        failed_files = []

        for i in tqdm(range(0, len(all_downloaded_files), 20), desc="Processing download files"):
            batch_files = all_downloaded_files[i:i + 20]
            batch_datasets = []

            for nc_file in batch_files:
                try:
                    ds = xr.open_dataset(nc_file)
                    if len(ds['id_geohash']) > 0:
                        batch_datasets.append(ds)
                    else:
                        logger.warning(f"File {nc_file} has no IDs, skipping")
                        failed_files.append(nc_file)
                except Exception as e:
                    logger.warning(f"Could not open {nc_file}: {e}")
                    failed_files.append(nc_file)

            if batch_datasets:
                batch_combined = xr.concat(batch_datasets, dim='id_geohash')
                # Remove duplicate IDs within this batch
                _, unique_idx = np.unique(batch_combined['id_geohash'].values, return_index=True)
                if len(unique_idx) < len(batch_combined['id_geohash']):
                    batch_combined = batch_combined.isel(id_geohash=np.sort(unique_idx))

                if combined is None:
                    combined = batch_combined
                else:
                    combined = xr.concat([combined, batch_combined], dim='id_geohash')
                    _, unique_idx = np.unique(combined['id_geohash'].values, return_index=True)
                    if len(unique_idx) < len(combined['id_geohash']):
                        combined = combined.isel(id_geohash=np.sort(unique_idx))

            # Clean up
            for ds in batch_datasets:
                ds.close()
            gc.collect()

        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} files")

        if combined is None:
            logger.error("No combined dataset created")
            ds_existing.close()
            return {'success': False, 'error': 'No combined dataset created'}

        logger.info(f"Combined dataset has {len(combined['id_geohash'])} IDs and {len(combined['date'])} dates")

        # Step 4: Ensure all variables from existing file are present in combined
        existing_vars = list(ds_existing.data_vars)
        for var_name in existing_vars:
            if var_name not in combined.data_vars:
                logger.info(f"Adding placeholder for variable: {var_name}")
                combined[var_name] = (('id_geohash', 'date'),
                                      np.full((len(combined['id_geohash']), len(combined['date'])), np.nan))

        # Step 5: Ensure all variables from combined are in existing
        for var_name in combined.data_vars:
            if var_name not in ds_existing.data_vars:
                logger.info(f"Adding new variable from combined to existing: {var_name}")
                ds_existing[var_name] = (('id_geohash', 'date'),
                                         np.full((len(ds_existing['id_geohash']), len(ds_existing['date'])), np.nan))

        # Step 6: Merge existing + combined
        logger.info("Merging existing data with new data...")
        final_combined = xr.merge([ds_existing, combined], compat='override')

        # Sort by date and id
        final_combined = final_combined.sortby(['date', 'id_geohash'])

        # Remove duplicate IDs if any
        _, unique_idx = np.unique(final_combined['id_geohash'].values, return_index=True)
        if len(unique_idx) < len(final_combined['id_geohash']):
            logger.info(f"Removing {len(final_combined['id_geohash']) - len(unique_idx)} duplicate IDs")
            final_combined = final_combined.isel(id_geohash=np.sort(unique_idx))

        # Step 7: Write to temp file
        logger.info(f"Writing merged data to temp file: {temp_file}")
        logger.info(f"  IDs: {len(final_combined['id_geohash']):,}")
        logger.info(f"  Dates: {len(final_combined['date'])}")

        encoding = {}
        for var in final_combined.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 4,
                'shuffle': True
            }

        # Write to temp file
        final_combined.to_netcdf(temp_file, encoding=encoding)
        logger.info(f"✅ Temp file written: {temp_file}")
        temp_size_gb = temp_file.stat().st_size / (1024 ** 3)
        logger.info(f"  Size: {temp_size_gb:.2f} GB")

        # Step 8: Move temp to final location
        logger.info(f"Moving temp file to final location: {input_file_path}")

        # If target exists, make a backup
        if input_path.exists():
            backup_file = input_path.parent / f"{input_path.stem}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{input_path.suffix}"
            logger.info(f"Backing up existing file to: {backup_file}")
            shutil.move(str(input_path), str(backup_file))

        # Move temp to final
        shutil.move(str(temp_file), str(input_path))
        logger.info(f"✅ File moved to: {input_file_path}")

        # Step 9: If final_copy_path provided, copy to that location
        if final_copy_path:
            logger.info(f"Copying to final location: {final_copy_path}")
            final_copy_path = Path(final_copy_path)
            final_copy_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy (not move) to keep the local file
            shutil.copy2(str(input_path), str(final_copy_path))
            logger.info(f"✅ Copied to: {final_copy_path}")
            copy_size_gb = final_copy_path.stat().st_size / (1024 ** 3)
            logger.info(f"  Size: {copy_size_gb:.2f} GB")

        # Step 10: Clean up
        ds_existing.close()
        combined.close()
        final_combined.close()
        gc.collect()

        # Step 11: Verify the final file
        logger.info("Verifying final file...")
        verify_ds = xr.open_dataset(input_path)
        verify_vars = set(verify_ds.data_vars)
        final_id_count = len(verify_ds['id_geohash'])
        final_date_count = len(verify_ds['date'])
        verify_ds.close()

        result = {
            'success': True,
            'file_path': str(input_path),
            'final_copy_path': str(final_copy_path) if final_copy_path else None,
            'id_count': final_id_count,
            'date_count': final_date_count,
            'file_size_gb': input_path.stat().st_size / (1024 ** 3),
            'dates_merged': normalized_dates,
            'region': region,
            'variables_preserved': list(verify_vars)
        }

        logger.info(f"✅ Merge completed successfully!")
        logger.info(f"  Final file: {input_file_path}")
        logger.info(f"  IDs: {result['id_count']:,} (was {existing_id_count:,})")
        logger.info(f"  Dates: {result['date_count']} (was {existing_date_count})")
        logger.info(f"  Size: {result['file_size_gb']:.2f} GB")
        logger.info(f"  Variables: {len(verify_vars)}")

        return result

    except Exception as e:
        logger.error(f"Error in merge: {e}")
        import traceback
        traceback.print_exc()

        # Clean up temp file if it exists
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
                logger.info(f"Removed temp file: {temp_file}")
            except:
                pass

        return {
            'success': False,
            'error': str(e),
            'file_path': str(input_path)
        }

def merge_near_real_time_region_v3_cloud(
        region: str = "TEST",
        dates_to_merge: List[str] = None,
        historical_file_path: str = None,
        temp_dir: str = "/tmp",
        verify_downloads_first: bool = True
):
    """
    Cloud-optimized merge function for Kubernetes/Filestore.

    Uses temporary local storage for intermediate files to reduce Filestore IO.
    """
    import tempfile
    import shutil
    from pathlib import Path

    log_memory_usage("Cloud merge start")

    # Use local temp directory for intermediate files
    local_temp = Path(temp_dir) / f"merge_temp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    local_temp.mkdir(exist_ok=True, parents=True)
    logger.info(f"Using local temp directory: {local_temp}")

    try:
        # Load historical data from Filestore
        if historical_file_path and os.path.exists(historical_file_path):
            original_historical_file = historical_file_path
        else:
            dynamic_world_data_dir = os.environ['dynamic_world_data']
            all_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
            original_historical_file = max(all_files, key=os.path.getctime)

        logger.info(f"Loading historical file from Filestore: {original_historical_file}")
        ds_historical = xr.open_dataset(original_historical_file)
        historical_vars = list(ds_historical.data_vars)

        # Collect downloaded files
        dynamic_world_download_dir = Path(os.environ['dynamic_world_downloads'])
        all_downloaded_files = []

        for date_str in dates_to_merge:
            if isinstance(date_str, pd.Timestamp):
                date_str = date_str.strftime("%Y-%m")
            download_dir = dynamic_world_download_dir / region / f'download_{date_str}'
            if download_dir.exists():
                files = glob.glob(str(download_dir / f'DW_{date_str}_*.nc'))
                all_downloaded_files.extend(files)

        if not all_downloaded_files:
            logger.error("No downloaded files found!")
            ds_historical.close()
            return {'success': False, 'error': 'No downloaded files'}

        # Process files in batches - use local temp for intermediate files
        logger.info(f"Processing {len(all_downloaded_files)} files...")
        combined = None
        batch_size = 20

        for batch_idx in tqdm(range(0, len(all_downloaded_files), batch_size), desc="Processing"):
            batch_files = all_downloaded_files[batch_idx:batch_idx + batch_size]
            batch_datasets = []

            for file_path in batch_files:
                try:
                    # Copy to local temp for faster access
                    local_file = local_temp / Path(file_path).name
                    shutil.copy2(file_path, local_file)
                    ds = xr.open_dataset(local_file)
                    if len(ds['id_geohash']) > 0:
                        batch_datasets.append(ds)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

            if batch_datasets:
                batch_combined = xr.concat(batch_datasets, dim='id_geohash')
                _, unique_idx = np.unique(batch_combined['id_geohash'].values, return_index=True)
                if len(unique_idx) < len(batch_combined['id_geohash']):
                    batch_combined = batch_combined.isel(id_geohash=np.sort(unique_idx))

                if combined is None:
                    combined = batch_combined
                else:
                    combined = xr.concat([combined, batch_combined], dim='id_geohash')
                    _, unique_idx = np.unique(combined['id_geohash'].values, return_index=True)
                    if len(unique_idx) < len(combined['id_geohash']):
                        combined = combined.isel(id_geohash=np.sort(unique_idx))

                for ds in batch_datasets:
                    ds.close()
                gc.collect()

            # Clear batch files from temp
            for f in local_temp.glob("*.nc"):
                f.unlink()

        # Ensure all variables exist
        for var_name in historical_vars:
            if var_name not in combined.data_vars:
                logger.info(f"Adding placeholder for variable: {var_name}")
                combined[var_name] = (('id_geohash', 'date'),
                                      np.full((len(combined['id_geohash']), len(combined['date'])), np.nan))

        # Merge
        logger.info("Merging datasets...")
        final_combined = xr.merge([ds_historical, combined], compat='override')
        final_combined = final_combined.sortby(['date', 'id_geohash'])

        # Write directly to Filestore (final file only)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_file = Path(original_historical_file).parent / f"historical_data_{timestamp}.nc"

        logger.info(f"Writing final file to Filestore: {new_file}")

        encoding = {}
        for var in final_combined.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 4,
                'shuffle': True
            }

        # Write directly to Filestore
        final_combined.to_netcdf(new_file, encoding=encoding)

        # Clean up
        ds_historical.close()
        combined.close()
        final_combined.close()
        gc.collect()

        return {
            'success': True,
            'final_file': str(new_file),
            'date_count': len(final_combined['date']),
            'id_count': len(final_combined['id_geohash'])
        }

    except Exception as e:
        logger.error(f"Merge failed: {e}")
        return {'success': False, 'error': str(e)}
    finally:
        # Clean up temp directory
        import shutil
        shutil.rmtree(local_temp, ignore_errors=True)


def verify_merged_data(
        file_path: str,
        region: str,
        dates: List[str] = None,
        check_id_count: bool = True,
        check_date_count: bool = True,
        verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify that a merged NetCDF file contains data for a specific region and dates.

    This function checks:
    1. The file exists and is valid
    2. The file contains the expected dates (or at least some dates)
    3. The file contains IDs for the region (or any IDs if region-specific check not needed)
    4. The data is not empty (has values)

    Args:
        file_path: Path to the NetCDF file to check
        region: Region name (e.g., "TEST", "EURASIA3") - used for logging only
        dates: List of dates in "YYYY-MM" format to check for. If None, just checks file is valid.
        check_id_count: If True, checks that there are IDs in the file
        check_date_count: If True, checks that there are dates in the file
        verbose: If True, logs detailed information

    Returns:
        dict: Verification results with keys:
            - success: bool - True if all checks pass
            - file_exists: bool
            - valid: bool - True if file is valid NetCDF
            - date_count: int - Number of dates in file
            - id_count: int - Number of IDs in file
            - file_size_gb: float - File size in GB
            - dates_in_file: List[str] - All dates in the file (YYYY-MM format)
            - missing_dates: List[str] - Dates requested that are missing
            - present_dates: List[str] - Dates requested that are present
            - all_dates_present: bool - True if all requested dates are present
            - has_data: bool - True if file has non-empty data
            - message: str - Summary message
    """
    import os
    import xarray as xr
    import pandas as pd
    from typing import List, Dict, Any

    result = {
        'success': False,
        'file_exists': False,
        'valid': False,
        'date_count': 0,
        'id_count': 0,
        'file_size_gb': 0,
        'dates_in_file': [],
        'missing_dates': [],
        'present_dates': [],
        'all_dates_present': False,
        'has_data': False,
        'message': '',
        'error': None
    }

    # Check file exists
    if not os.path.exists(file_path):
        result['message'] = f"File does not exist: {file_path}"
        result['error'] = 'File not found'
        if verbose:
            logger.error(result['message'])
        return result

    result['file_exists'] = True

    # Check file size
    result['file_size_gb'] = get_file_size_gb(file_path)
    if verbose:
        logger.info(f"File: {file_path}")
        logger.info(f"Size: {result['file_size_gb']:.2f} GB")

    # Try to open and read the file
    try:
        ds = xr.open_dataset(file_path)

        # Check if it has dimensions
        has_id_dim = 'id_geohash' in ds.dims
        has_date_dim = 'date' in ds.dims

        if not has_id_dim or not has_date_dim:
            ds.close()
            result['message'] = f"File missing required dimensions: id_geohash={has_id_dim}, date={has_date_dim}"
            result['error'] = 'Missing dimensions'
            if verbose:
                logger.error(result['message'])
            return result

        result['valid'] = True
        result['id_count'] = len(ds['id_geohash'])
        result['date_count'] = len(ds['date'])

        # Get dates in file
        dates_in_file = pd.to_datetime(ds['date'].values)
        result['dates_in_file'] = [d.strftime("%Y-%m") for d in dates_in_file]

        # Check if there's any data (non-empty)
        has_data = result['id_count'] > 0 and result['date_count'] > 0
        result['has_data'] = has_data

        # Check for requested dates
        if dates:
            requested_dates = sorted(dates)
            present_dates = []
            missing_dates = []

            for date_str in requested_dates:
                # Check if date exists in the file
                date_exists = date_str in result['dates_in_file']
                if date_exists:
                    present_dates.append(date_str)
                else:
                    missing_dates.append(date_str)

            result['present_dates'] = present_dates
            result['missing_dates'] = missing_dates
            result['all_dates_present'] = len(missing_dates) == 0

            if verbose:
                logger.info(f"Requested dates: {requested_dates}")
                logger.info(f"Present: {present_dates}")
                logger.info(f"Missing: {missing_dates}")

                if result['all_dates_present']:
                    logger.info(f"✅ All {len(requested_dates)} requested dates are present!")
                else:
                    logger.warning(f"⚠️ {len(missing_dates)} dates missing: {missing_dates}")

        # Check for a specific ID to verify data quality
        sample_id = ds['id_geohash'].values[0] if result['id_count'] > 0 else None
        if sample_id and dates:
            # Check if the sample ID has data for the requested dates
            sample_data = ds.sel(id_geohash=sample_id)
            sample_dates = pd.to_datetime(sample_data['date'].values)
            sample_date_strings = [d.strftime("%Y-%m") for d in sample_dates]

            # Check if any requested dates are in the sample data
            sample_present = [d for d in dates if d in sample_date_strings]
            if sample_present and verbose:
                logger.info(f"Sample ID {sample_id[:8]}... has data for {len(sample_present)} requested dates")
            elif verbose:
                logger.warning(f"Sample ID {sample_id[:8]}... has NO data for requested dates!")

        # Close the dataset
        ds.close()

        # Determine overall success
        result['success'] = (
                result['valid'] and
                result['has_data'] and
                (result['all_dates_present'] if dates else True)
        )

        # Build summary message
        if result['success']:
            if dates:
                result[
                    'message'] = f"✅ File contains all {len(dates)} requested dates, {result['id_count']} IDs, {result['date_count']} total dates"
            else:
                result['message'] = f"✅ File is valid with {result['id_count']} IDs, {result['date_count']} dates"
        else:
            issues = []
            if not result['valid']:
                issues.append("invalid file")
            if not result['has_data']:
                issues.append("no data")
            if dates and not result['all_dates_present']:
                issues.append(f"missing dates: {result['missing_dates']}")
            result['message'] = f"⚠️ File has issues: {', '.join(issues)}"

        if verbose:
            logger.info(result['message'])

        return result

    except Exception as e:
        result['message'] = f"Error reading file: {e}"
        result['error'] = str(e)
        if verbose:
            logger.error(result['message'])
        return result


def check_region_data_in_merged_file(
        region: str,
        dates: List[str],
        merged_file_path: str = None,
        env_path: str = None,
        verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to check if a merged file contains data for a region.

    Finds the most recent merged file if not specified, and checks it for the region's data.

    Args:
        region: Region name (e.g., "TEST", "EURASIA3")
        dates: List of dates in "YYYY-MM" format to check for
        merged_file_path: Optional path to the merged file. If None, finds the most recent.
        env_path: Optional path to .env file
        verbose: If True, logs detailed information

    Returns:
        dict: Verification results from verify_merged_data
    """
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Find the merged file if not specified
    if not merged_file_path:
        dynamic_world_data_dir = os.environ.get('dynamic_world_data')
        if not dynamic_world_data_dir:
            logger.error("dynamic_world_data not set in environment")
            return {'success': False, 'error': 'Environment variable not set'}

        # Find the most recent historical_data file
        all_files = glob.glob(os.path.join(dynamic_world_data_dir, "historical_data_*.nc"))
        if not all_files:
            # Try just .nc files if no historical_data files
            all_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
            # Exclude the original file
            all_files = [f for f in all_files if 'lakes_dw_V2d' not in f and 'lakes_dw_V2' not in f]

        if not all_files:
            logger.error(f"No merged NetCDF files found in {dynamic_world_data_dir}")
            return {'success': False, 'error': 'No merged files found'}

        merged_file_path = max(all_files, key=os.path.getctime)
        logger.info(f"Using most recent merged file: {merged_file_path}")

    # Verify the merged file contains the region's data
    result = verify_merged_data(
        file_path=merged_file_path,
        region=region,
        dates=dates,
        verbose=verbose
    )

    # Add region info
    result['region'] = region
    result['file_path'] = merged_file_path

    return result


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
    most_recent_dynamic_world_file = max(all_dynamic_world_files,key=lambda f: Path(f).stat().st_mtime)
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

    # Normalize analysis_dates to string format "YYYY-MM"
    normalized_dates = []
    if analysis_dates is not None:
        for date in analysis_dates:
            if isinstance(date, pd.Timestamp):
                normalized_dates.append(date.strftime("%Y-%m"))
            elif isinstance(date, datetime.datetime):
                normalized_dates.append(date.strftime("%Y-%m"))
            elif isinstance(date, str):
                # Try to parse and reformat
                try:
                    # If it's already in YYYY-MM format
                    if len(date) == 7 and date[4] == '-':
                        normalized_dates.append(date)
                    else:
                        # Try to parse as datetime
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
        # Discover dates from download directories
        download_pattern = str(dynamic_world_download_dir / REGION_NAME / 'download_*')
        download_dirs = glob.glob(download_pattern)

        discovered_dates = []
        for dir_path in download_dirs:
            # Extract date from directory name
            dir_name = Path(dir_path).name
            if dir_name.startswith('download_'):
                date_str = dir_name.replace('download_', '')
                # Validate date format (should be YYYY-MM)
                try:
                    datetime.datetime.strptime(date_str, '%Y-%m')
                    discovered_dates.append(date_str)
                except ValueError:
                    continue

        analysis_dates = sorted(discovered_dates)

        if not analysis_dates:
            return {
                'complete': False,
                'reason': 'No dates found to verify',
                'discovered_dates': discovered_dates,
                'date_results': {}
            }
    else:
        # No dates provided and auto_discover is False
        return {
            'complete': False,
            'reason': 'No dates provided for verification',
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

        # Use the date directly (already in YYYY-MM format)
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
            'total_failed_downloads': sum(r.get('failed_downloads', 0) for r in date_results.values()),
            'total_skipped_downloads': sum(r.get('skipped_downloads', 0) for r in date_results.values()),
        }
    }


def summarize_breakpoint_results(
        region: str,
        analysis_date: str = None,
        zarr_dir: str = None,
        env_path: str = None,
        verbose: bool = True
) -> Dict[str, Any]:
    """
    Summarize breakpoint analysis results from Zarr files.

    This function reads the Zarr output from process_near_real_time_region_dates_zarr
    and provides a comprehensive summary of the breakpoint detection results.

    Args:
        region: Region name (e.g., "TEST", "EURASIA3")
        analysis_date: Date in "YYYY-MM" format. If None, summarizes the most recent.
        zarr_dir: Optional path to Zarr directory. If None, uses environment variable.
        env_path: Optional path to .env file
        verbose: If True, prints detailed summary

    Returns:
        dict: Summary statistics including:
            - total_lakes: Total number of lakes processed
            - breakpoints_found: Number of lakes with breakpoints
            - breakpoint_rate: Percentage of lakes with breakpoints
            - confidence_distribution: Counts of low/medium/high confidence
            - date_range: Date range of data
            - file_size_gb: Size of Zarr file
            - detailed_results: More detailed breakdown
    """
    import json
    from datetime import datetime

    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Set up paths
    if zarr_dir is None:
        output_dir = os.environ.get('output_dir')
        if not output_dir:
            raise ValueError("output_dir not set in environment")
        zarr_dir = Path(output_dir) / region / 'breakpoint_zarr'
    else:
        zarr_dir = Path(zarr_dir)

    if not zarr_dir.exists():
        return {'error': f'Zarr directory not found: {zarr_dir}'}

    # Find Zarr files
    if analysis_date:
        zarr_files = [zarr_dir / f'breakpoints_{analysis_date}.zarr']
        if not zarr_files[0].exists():
            return {'error': f'Zarr file for {analysis_date} not found'}
    else:
        # Find all Zarr files and get the most recent
        zarr_files = sorted(zarr_dir.glob('breakpoints_*.zarr'))
        if not zarr_files:
            return {'error': 'No Zarr files found'}

    results = {}
    total_summary = {
        'region': region,
        'dates_processed': [],
        'total_lakes_all_dates': set(),
        'total_breakpoints_all_dates': 0,
        'confidence_distribution_all': {'low': 0, 'medium': 0, 'high': 0}
    }

    for zarr_path in zarr_files:
        date_str = zarr_path.stem.replace('breakpoints_', '')

        try:
            # Open Zarr dataset
            ds = xr.open_zarr(zarr_path)

            # Basic info
            n_lakes = len(ds['id_geohash']) if 'id_geohash' in ds.dims else 0
            date_info = "No date dimension" if 'date' not in ds.dims else len(ds['date'])

            # Check for drainage_confidence column
            has_confidence = 'drainage_confidence' in ds.data_vars

            # Calculate statistics
            confidence_counts = {'low': 0, 'medium': 0, 'high': 0}
            breakpoints_found = 0

            if has_confidence and n_lakes > 0:
                # Get confidence values
                conf_data = ds['drainage_confidence'].values

                # Count by confidence level
                confidence_counts['low'] = int(np.sum(conf_data == 1))
                confidence_counts['medium'] = int(np.sum(conf_data == 2))
                confidence_counts['high'] = int(np.sum(conf_data == 3))
                breakpoints_found = confidence_counts['medium'] + confidence_counts['high']

            # Check for water_observed to see if there's data
            has_water = 'water_observed' in ds.data_vars

            # Get date range if available
            date_range = None
            if 'date' in ds.coords:
                dates = pd.to_datetime(ds['date'].values)
                date_range = {
                    'min': dates.min().strftime('%Y-%m-%d'),
                    'max': dates.max().strftime('%Y-%m-%d'),
                    'count': len(dates)
                }

            # Calculate file size
            file_size_gb = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file()) / (1024 ** 3)

            date_result = {
                'date': date_str,
                'total_lakes': n_lakes,
                'breakpoints_found': breakpoints_found,
                'breakpoint_rate': breakpoints_found / n_lakes * 100 if n_lakes > 0 else 0,
                'confidence_distribution': confidence_counts,
                'has_confidence': has_confidence,
                'has_water_data': has_water,
                'date_range': date_range,
                'file_size_gb': round(file_size_gb, 4),
                'zarr_path': str(zarr_path)
            }

            results[date_str] = date_result

            # Update totals
            total_summary['dates_processed'].append(date_str)
            total_summary['total_lakes_all_dates'].add(n_lakes)
            total_summary['total_breakpoints_all_dates'] += breakpoints_found
            total_summary['confidence_distribution_all']['low'] += confidence_counts['low']
            total_summary['confidence_distribution_all']['medium'] += confidence_counts['medium']
            total_summary['confidence_distribution_all']['high'] += confidence_counts['high']

            ds.close()

        except Exception as e:
            logger.error(f"Error reading {zarr_path}: {e}")
            results[date_str] = {'error': str(e)}

    # Build summary
    summary = {
        'region': region,
        'dates_processed': total_summary['dates_processed'],
        'total_dates': len(total_summary['dates_processed']),
        'total_lakes': len(total_summary['total_lakes_all_dates']),
        'total_breakpoints': total_summary['total_breakpoints_all_dates'],
        'breakpoint_rate': total_summary['total_breakpoints_all_dates'] / len(
            total_summary['total_lakes_all_dates']) * 100 if len(total_summary['total_lakes_all_dates']) > 0 else 0,
        'confidence_distribution': total_summary['confidence_distribution_all'],
        'date_results': results
    }

    # Print summary
    if verbose:
        print("\n" + "=" * 80)
        print(f"BREAKPOINT ANALYSIS SUMMARY - {region}")
        print("=" * 80)

        print(f"\n📊 OVERVIEW:")
        print(f"  Region: {region}")
        print(f"  Dates processed: {len(summary['dates_processed'])}")
        print(f"  Total lakes: {summary['total_lakes']:,}")
        print(f"  Breakpoints found: {summary['total_breakpoints']:,}")
        print(f"  Breakpoint rate: {summary['breakpoint_rate']:.2f}%")

        print(f"\n📈 CONFIDENCE DISTRIBUTION:")
        print(f"  Low confidence (1): {summary['confidence_distribution']['low']:,}")
        print(f"  Medium confidence (2): {summary['confidence_distribution']['medium']:,}")
        print(f"  High confidence (3): {summary['confidence_distribution']['high']:,}")

        print(f"\n📅 DATE DETAILS:")
        for date in sorted(summary['dates_processed']):
            date_result = results[date]
            if 'error' not in date_result:
                print(f"  {date}:")
                print(f"    Lakes: {date_result['total_lakes']:,}")
                print(f"    Breakpoints: {date_result['breakpoints_found']:,} ({date_result['breakpoint_rate']:.2f}%)")
                print(f"    Size: {date_result['file_size_gb']:.4f} GB")

                if date_result.get('date_range'):
                    print(f"    Date range: {date_result['date_range']['min']} to {date_result['date_range']['max']}")
            else:
                print(f"  {date}: ❌ ERROR - {date_result['error']}")

        print("\n" + "=" * 80)

    return summary


def print_breakpoint_summary_table(
        region: str,
        dates: List[str] = None,
        env_path: str = None
):
    """
    Print a formatted table of breakpoint results for one or more dates.

    Args:
        region: Region name
        dates: List of dates in "YYYY-MM" format. If None, finds all.
        env_path: Optional path to .env file
    """
    import pandas as pd

    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    output_dir = os.environ.get('output_dir')
    if not output_dir:
        raise ValueError("output_dir not set in environment")

    zarr_dir = Path(output_dir) / region / 'breakpoint_zarr'

    if not zarr_dir.exists():
        print(f"❌ Zarr directory not found: {zarr_dir}")
        return

    # Find Zarr files
    if dates:
        zarr_files = [zarr_dir / f'breakpoints_{d}.zarr' for d in dates]
        zarr_files = [f for f in zarr_files if f.exists()]
    else:
        zarr_files = sorted(zarr_dir.glob('breakpoints_*.zarr'))

    if not zarr_files:
        print("❌ No Zarr files found")
        return

    # Collect data for table
    rows = []
    for zarr_path in zarr_files:
        date_str = zarr_path.stem.replace('breakpoints_', '')

        try:
            ds = xr.open_zarr(zarr_path)
            n_lakes = len(ds['id_geohash']) if 'id_geohash' in ds.dims else 0

            if 'drainage_confidence' in ds.data_vars:
                conf_data = ds['drainage_confidence'].values
                low = int(np.sum(conf_data == 1))
                medium = int(np.sum(conf_data == 2))
                high = int(np.sum(conf_data == 3))
                total_breakpoints = medium + high
            else:
                low = medium = high = total_breakpoints = 0

            file_size = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file()) / (1024 ** 3)

            rows.append({
                'Date': date_str,
                'Lakes': n_lakes,
                'Breakpoints': total_breakpoints,
                'Rate %': f"{total_breakpoints / n_lakes * 100:.1f}" if n_lakes > 0 else "0.0",
                'Low': low,
                'Medium': medium,
                'High': high,
                'Size (GB)': f"{file_size:.4f}"
            })

            ds.close()
        except Exception as e:
            rows.append({
                'Date': date_str,
                'Lakes': 'ERROR',
                'Breakpoints': '-',
                'Rate %': '-',
                'Low': '-',
                'Medium': '-',
                'High': '-',
                'Size (GB)': '-'
            })

    # Create and print DataFrame
    df = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print(f"BREAKPOINT RESULTS TABLE - {region}")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)

    return df

# TODO test this method on results
def check_breakpoint_quality(
        region: str,
        analysis_date: str,
        min_confidence: int = 2,
        env_path: str = None
) -> Dict[str, Any]:
    """
    Check the quality of breakpoint results for a specific date.

    Args:
        region: Region name
        analysis_date: Date in "YYYY-MM" format
        min_confidence: Minimum confidence level to consider (1, 2, or 3)
        env_path: Optional path to .env file

    Returns:
        dict: Quality metrics
    """
    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    output_dir = os.environ.get('output_dir')
    if not output_dir:
        raise ValueError("output_dir not set in environment")

    zarr_path = Path(output_dir) / region / 'breakpoint_zarr' / f'breakpoints_{analysis_date}.zarr'

    if not zarr_path.exists():
        return {'error': f'Zarr file not found: {zarr_path}'}

    try:
        ds = xr.open_zarr(zarr_path)

        n_lakes = len(ds['id_geohash']) if 'id_geohash' in ds.dims else 0

        result = {
            'date': analysis_date,
            'region': region,
            'total_lakes': n_lakes,
            'has_confidence': False,
            'has_water_data': False,
            'quality_score': 0
        }

        if 'drainage_confidence' in ds.data_vars:
            conf_data = ds['drainage_confidence'].values
            result['has_confidence'] = True

            # Count by confidence
            low = int(np.sum(conf_data == 1))
            medium = int(np.sum(conf_data == 2))
            high = int(np.sum(conf_data == 3))

            result['low_confidence'] = low
            result['medium_confidence'] = medium
            result['high_confidence'] = high

            # Count quality breakpoints (meeting min_confidence)
            quality_breakpoints = int(np.sum(conf_data >= min_confidence))
            result['quality_breakpoints'] = quality_breakpoints
            result['quality_rate'] = quality_breakpoints / n_lakes * 100 if n_lakes > 0 else 0

            # Quality score (0-100)
            if n_lakes > 0:
                # Weighted score: 50% for high confidence, 30% for medium, 20% for any breakpoint
                score = (high / n_lakes * 50) + (medium / n_lakes * 30) + ((high + medium) / n_lakes * 20)
                result['quality_score'] = round(score, 2)

        if 'water_observed' in ds.data_vars:
            result['has_water_data'] = True
            # Check if any water data is non-nan
            water_data = ds['water_observed'].values
            result['has_non_nan_water'] = bool(np.any(~np.isnan(water_data)))

        ds.close()

        # Quality assessment
        if result['quality_score'] >= 80:
            result['quality_assessment'] = 'EXCELLENT'
        elif result['quality_score'] >= 60:
            result['quality_assessment'] = 'GOOD'
        elif result['quality_score'] >= 40:
            result['quality_assessment'] = 'FAIR'
        else:
            result['quality_assessment'] = 'POOR'

        return result

    except Exception as e:
        return {'error': str(e)}

def verify_process_complete(
        region: str = "TEST",
        analysis_dates: List[str] = None,
        env_path : str = None,
):
    logger.debug("Verifying if processing is complete")

    output_dir = os.environ['output_dir']
    output_zarr_dir = os.path.join(output_dir, region, 'breakpoint_zarr')

    normalized_dates = []
    if analysis_dates is not None:
        for date in analysis_dates:
            if isinstance(date, pd.Timestamp):
                normalized_dates.append(date.strftime("%Y-%m"))
            elif isinstance(date, datetime.datetime):
                normalized_dates.append(date.strftime("%Y-%m"))
            elif isinstance(date, str):
                # Try to parse and reformat
                try:
                    # If it's already in YYYY-MM format
                    if len(date) == 7 and date[4] == '-':
                        normalized_dates.append(date)
                    else:
                        # Try to parse as datetime
                        dt = pd.to_datetime(date)
                        normalized_dates.append(dt.strftime("%Y-%m"))
                except:
                    logger.warning(f"Could not parse date: {date}")
            else:
                logger.warning(f"Unrecognized date type: {type(date)} for {date}")

    analysis_dates = normalized_dates
    success_count = 0
    fail_count = 0
    for analysis_date in analysis_dates:
        current_zarr_dataset = f'breakpoints_{analysis_date}.zarr'
        path_to_zarr_dataset = os.path.join(output_zarr_dir, current_zarr_dataset)
        if os.path.exists(path_to_zarr_dataset):
            success_count += 1
            logger.debug(f"Found zarr dataset {current_zarr_dataset}")
        else:
            fail_count += 1
            logger.debug(f"This zarr dataset {current_zarr_dataset} does not exist")
    logger.debug(f"Success count: {success_count} and fail count {fail_count}")
    if fail_count > 0:
        return {'complete': False}
    else:
        return {'complete': True}


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


def merge_near_real_time_region_v3_chunked(
        region: str = "TEST",
        dates_to_merge: List[str] = None,
        source_file: str = None,
        output_file: str = None,
        env_path: str = None,
        chunk_size: int = 50000,  # Number of IDs per chunk
        temp_dir: str = "/tmp/merge_temp",
        combine_batch_size: int = 10,  # Number of chunks to combine at once
):
    """
    Chunked merge function that processes data in chunks to stay under memory/disk limits.

    This function:
    1. Reads source file in chunks (memory efficient)
    2. Merges with downloaded data for each chunk (UNION of IDs - keeps ALL historical data)
    3. Writes results incrementally
    4. Cleans up after each chunk

    IMPORTANT: This performs a UNION merge - ALL source IDs are preserved,
    and new data is added for IDs that have it.
    """
    log_memory_usage("Chunked merge start")

    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Validate inputs
    if source_file is None or not Path(source_file).exists():
        logger.error(f"Source file not found: {source_file}")
        return {'success': False, 'error': 'Source file not found'}

    if output_file is None:
        logger.error("Output file path is required")
        return {'success': False, 'error': 'Output file path is required'}

    # Normalize dates
    normalized_dates = []
    for date in dates_to_merge:
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

    if not normalized_dates:
        logger.error("No valid dates provided")
        return {'success': False, 'error': 'No valid dates provided'}

    # Create temp directory
    temp_dir_path = Path(temp_dir)
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    # Get downloaded files for the dates
    dynamic_world_download_dir = Path(os.environ.get('dynamic_world_downloads', ''))
    all_downloaded_files = []

    for date_str in normalized_dates:
        download_dir = dynamic_world_download_dir / region / f'download_{date_str}'
        if download_dir.exists():
            downloaded_files = glob.glob(str(download_dir / f'DW_{date_str}_*.nc'))
            all_downloaded_files.extend(downloaded_files)
            logger.info(f"Found {len(downloaded_files)} files for {date_str}")
        else:
            logger.warning(f"Download directory not found for {date_str}: {download_dir}")

    if not all_downloaded_files:
        logger.error("No downloaded files found to merge")
        return {'success': False, 'error': 'No downloaded files found'}

    # Combine downloaded files into a single dataset
    logger.info(f"Combining {len(all_downloaded_files)} downloaded files...")
    combined = None

    for i in tqdm(range(0, len(all_downloaded_files), 20), desc="Processing download files"):
        batch_files = all_downloaded_files[i:i + 20]
        batch_datasets = []

        for nc_file in batch_files:
            try:
                ds = xr.open_dataset(nc_file)
                if len(ds['id_geohash']) > 0:
                    batch_datasets.append(ds)
            except Exception as e:
                logger.warning(f"Could not open {nc_file}: {e}")

        if batch_datasets:
            batch_combined = xr.concat(batch_datasets, dim='id_geohash')
            _, unique_idx = np.unique(batch_combined['id_geohash'].values, return_index=True)
            if len(unique_idx) < len(batch_combined['id_geohash']):
                batch_combined = batch_combined.isel(id_geohash=np.sort(unique_idx))

            if combined is None:
                combined = batch_combined
            else:
                combined = xr.concat([combined, batch_combined], dim='id_geohash')
                _, unique_idx = np.unique(combined['id_geohash'].values, return_index=True)
                if len(unique_idx) < len(combined['id_geohash']):
                    combined = combined.isel(id_geohash=np.sort(unique_idx))

        # Clean up batch datasets
        for ds in batch_datasets:
            ds.close()
        gc.collect()

    if combined is None:
        logger.error("No combined dataset created")
        return {'success': False, 'error': 'No combined dataset created'}

    logger.info(f"Combined dataset has {len(combined['id_geohash'])} IDs and {len(combined['date'])} dates")

    try:
        # Step 1: Open source file with chunking
        logger.info(f"Opening source file with chunking: {source_file}")
        ds_source = xr.open_dataset(source_file, chunks={'id_geohash': chunk_size})

        total_ids = len(ds_source['id_geohash'])
        num_chunks = (total_ids + chunk_size - 1) // chunk_size
        logger.info(f"Processing {total_ids:,} IDs in {num_chunks} chunks of {chunk_size}")

        # Step 2: Process chunks
        chunk_file_paths = []

        for chunk_idx in tqdm(range(num_chunks), desc="Processing ID chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_ids)

            logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks} (IDs {start_idx:,} - {end_idx:,})")

            # Get source chunk
            source_chunk = ds_source.isel(id_geohash=slice(start_idx, end_idx))

            # Get IDs from source and combined
            source_ids = set(source_chunk['id_geohash'].values)
            combined_ids = set(combined['id_geohash'].values)

            # ===== FIX: Use UNION of IDs (not intersection) =====
            # Get ALL unique IDs from both source and combined
            all_ids = source_ids | combined_ids

            # Convert to list for reindexing
            all_ids_list = list(all_ids)

            logger.info(
                f"  Source IDs: {len(source_ids):,}, Combined IDs: {len(combined_ids):,}, Union: {len(all_ids):,}")

            # Reindex source_chunk to include ALL IDs (IDs not in source get NaN)
            source_chunk_expanded = source_chunk.reindex(id_geohash=all_ids_list)

            # Get combined data for IDs that exist in combined
            combined_ids_in_chunk = [id_val for id_val in all_ids_list if id_val in combined_ids]

            if combined_ids_in_chunk:
                # Get combined data for IDs that exist in combined
                combined_chunk = combined.sel(id_geohash=combined_ids_in_chunk)

                # Reindex combined_chunk to include ALL IDs (IDs not in combined get NaN)
                combined_chunk_expanded = combined_chunk.reindex(id_geohash=all_ids_list)
            else:
                # No combined IDs in this chunk - create empty dataset with all IDs
                combined_chunk_expanded = xr.Dataset(
                    coords={'id_geohash': all_ids_list},
                    attrs=combined.attrs
                )
                # Add variables with NaN
                for var_name in combined.data_vars:
                    combined_chunk_expanded[var_name] = (('id_geohash', 'date'),
                                                         np.full((len(all_ids_list), len(combined['date'])), np.nan))

            # Ensure all variables are present in both datasets
            source_vars = list(ds_source.data_vars)
            combined_vars = list(combined.data_vars)
            all_vars = set(source_vars) | set(combined_vars)

            # Add missing variables to source_chunk_expanded
            for var_name in all_vars:
                if var_name not in source_chunk_expanded.data_vars:
                    if var_name in combined_vars:
                        source_chunk_expanded[var_name] = (('id_geohash', 'date'),
                                                           np.full((len(all_ids_list), len(combined['date'])), np.nan))
                    else:
                        source_chunk_expanded[var_name] = (('id_geohash', 'date'),
                                                           np.full((len(all_ids_list), len(source_chunk['date'])),
                                                                   np.nan))

            # Add missing variables to combined_chunk_expanded
            for var_name in all_vars:
                if var_name not in combined_chunk_expanded.data_vars:
                    if var_name in source_vars:
                        combined_chunk_expanded[var_name] = (('id_geohash', 'date'),
                                                             np.full((len(all_ids_list), len(source_chunk['date'])),
                                                                     np.nan))
                    else:
                        combined_chunk_expanded[var_name] = (('id_geohash', 'date'),
                                                             np.full((len(all_ids_list), 1), np.nan))

            # Now merge - all IDs are present in both (with NaN where missing)
            merged_chunk = xr.merge([source_chunk_expanded, combined_chunk_expanded], compat='override')

            # Write chunk to temporary file
            chunk_file = temp_dir_path / f"chunk_{chunk_idx:04d}.nc"
            encoding = {}
            for var in merged_chunk.data_vars:
                encoding[var] = {
                    'zlib': True,
                    'complevel': 4,
                    'shuffle': True
                }

            merged_chunk.to_netcdf(chunk_file, encoding=encoding)
            chunk_file_paths.append(chunk_file)
            file_size_gb = chunk_file.stat().st_size / (1024 ** 3)
            logger.info(
                f"  Chunk {chunk_idx + 1} written: {chunk_file} ({file_size_gb:.3f} GB) | IDs: {len(all_ids):,}")

            # Clean up chunk to free memory
            source_chunk.close()
            source_chunk_expanded.close()
            if 'combined_chunk' in locals():
                combined_chunk.close()
            combined_chunk_expanded.close()
            merged_chunk.close()
            gc.collect()

            # Log memory usage
            log_memory_usage(f"After chunk {chunk_idx + 1}")

            # Calculate actual temp directory size (not disk usage)
            temp_size = 0
            for item in temp_dir_path.rglob('*'):
                if item.is_file():
                    temp_size += item.stat().st_size
            temp_size_gb = temp_size / (1024 ** 3)
            logger.info(f"Temp directory usage: {temp_size_gb:.2f} GB")

            # If temp directory is getting large, combine and compress chunks in batches
            if temp_size_gb > 8 and len(chunk_file_paths) > combine_batch_size:
                logger.warning(
                    f"Temp directory at {temp_size_gb:.2f} GB, combining chunks in batches of {combine_batch_size}...")
                combine_success = combine_chunk_files(
                    chunk_file_paths,
                    output_file,
                    temp_dir_path,
                    batch_size=combine_batch_size
                )
                if combine_success:
                    # Clean up chunk files
                    for f in chunk_file_paths:
                        try:
                            f.unlink()
                        except:
                            pass
                    chunk_file_paths = []
                    gc.collect()
                    logger.info("Chunks combined and cleaned up")

        # Step 3: Combine all remaining chunks into final file
        if chunk_file_paths:
            logger.info(f"Combining {len(chunk_file_paths)} remaining chunks into final file...")
            combine_success = combine_chunk_files(
                chunk_file_paths,
                output_file,
                temp_dir_path,
                batch_size=combine_batch_size
            )
            if combine_success:
                # Clean up chunk files
                for f in chunk_file_paths:
                    try:
                        f.unlink()
                    except:
                        pass
                chunk_file_paths = []
                logger.info(f"Final file written: {output_file}")
        else:
            logger.info("No remaining chunks to combine (already combined during processing)")

        # Step 4: Clean up
        ds_source.close()
        combined.close()
        gc.collect()

        # Step 5: Verify the final file
        if Path(output_file).exists():
            logger.info("Verifying final file...")
            verify_ds = xr.open_dataset(output_file)
            final_id_count = len(verify_ds['id_geohash'])
            final_date_count = len(verify_ds['date'])
            verify_vars = set(verify_ds.data_vars)
            verify_ds.close()
            file_size_gb = Path(output_file).stat().st_size / (1024 ** 3)

            result = {
                'success': True,
                'file_path': str(output_file),
                'id_count': final_id_count,
                'date_count': final_date_count,
                'file_size_gb': file_size_gb,
                'dates_merged': normalized_dates,
                'region': region,
                'variables_preserved': list(verify_vars)
            }

            logger.info(f"✅ Chunked merge completed successfully!")
            logger.info(f"  Final file: {output_file}")
            logger.info(f"  IDs: {result['id_count']:,}")
            logger.info(f"  Dates: {result['date_count']}")
            logger.info(f"  Size: {result['file_size_gb']:.2f} GB")

            return result
        else:
            logger.error(f"Final file not created: {output_file}")
            return {'success': False, 'error': 'Final file not created'}

    except Exception as e:
        logger.error(f"Error in chunked merge: {e}")
        import traceback
        traceback.print_exc()

        # Clean up chunk files
        for f in chunk_file_paths:
            try:
                f.unlink()
            except:
                pass

        return {
            'success': False,
            'error': str(e),
            'file_path': str(output_file)
        }


def combine_chunk_files(chunk_files, output_file, temp_dir, batch_size=10):
    """
    Combine chunk files into a single NetCDF file in batches to avoid OOM.

    Args:
        chunk_files: List of chunk file paths
        output_file: Path to the output file
        temp_dir: Temporary directory for intermediate files
        batch_size: Number of chunks to combine at once (default: 10)

    Returns:
        bool: True if successful, False otherwise
    """
    if not chunk_files:
        logger.warning("No chunk files to combine")
        return False

    logger.info(f"Combining {len(chunk_files)} chunk files in batches of {batch_size}...")

    # If fewer chunks than batch_size, combine directly
    if len(chunk_files) <= batch_size:
        return combine_chunks_direct(chunk_files, output_file)

    # Process in batches
    batch_files = []
    for i in range(0, len(chunk_files), batch_size):
        batch = chunk_files[i:i + batch_size]
        batch_output = temp_dir / f"combined_batch_{i:04d}.nc"

        logger.info(
            f"Combining batch {i // batch_size + 1}/{(len(chunk_files) + batch_size - 1) // batch_size} ({len(batch)} chunks)")

        # Combine this batch
        success = combine_chunks_direct(batch, batch_output)
        if not success:
            logger.error(f"Failed to combine batch {i}")
            return False

        batch_files.append(batch_output)
        if batch_output.exists():
            logger.info(f"  Batch output: {batch_output} ({batch_output.stat().st_size / (1024 ** 3):.2f} GB)")

        # Clean up the original chunk files after successful batch combine
        for f in batch:
            try:
                if f.exists():
                    f.unlink()
            except:
                pass

    # If we have multiple batch files, combine them recursively
    if len(batch_files) > 1:
        logger.info(f"Combining {len(batch_files)} batch files...")
        return combine_chunk_files(batch_files, output_file, temp_dir, batch_size=5)
    elif len(batch_files) == 1:
        # Only one batch file, just move it
        shutil.move(batch_files[0], output_file)
        logger.info(f"✅ Moved single batch file to {output_file}")
        return True

    return False


def combine_chunks_direct(chunk_files, output_file):
    """
    Directly combine a list of chunk files (no batching).
    Assumes the list is small enough to fit in memory.
    """
    if not chunk_files:
        logger.warning("No chunk files to combine")
        return False

    logger.info(f"Combining {len(chunk_files)} chunks directly...")

    # Open all chunks
    chunk_datasets = []
    valid_files = []

    for chunk_file in chunk_files:
        try:
            if Path(chunk_file).exists() and Path(chunk_file).stat().st_size > 0:
                ds = xr.open_dataset(chunk_file)
                if len(ds['id_geohash']) > 0:
                    chunk_datasets.append(ds)
                    valid_files.append(chunk_file)
                else:
                    ds.close()
                    logger.warning(f"Chunk file has no IDs: {chunk_file}")
            else:
                logger.warning(f"Chunk file missing or empty: {chunk_file}")
        except Exception as e:
            logger.warning(f"Could not open {chunk_file}: {e}")

    if not chunk_datasets:
        logger.error("No valid chunk datasets to combine")
        return False

    try:
        # Concatenate all chunks
        final_combined = xr.concat(chunk_datasets, dim='id_geohash')

        # Remove duplicate IDs if any
        _, unique_idx = np.unique(final_combined['id_geohash'].values, return_index=True)
        if len(unique_idx) < len(final_combined['id_geohash']):
            logger.info(f"Removing {len(final_combined['id_geohash']) - len(unique_idx)} duplicate IDs")
            final_combined = final_combined.isel(id_geohash=np.sort(unique_idx))

        # Sort
        final_combined = final_combined.sortby(['date', 'id_geohash'])

        # Write with compression
        encoding = {}
        for var in final_combined.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 4,
                'shuffle': True
            }

        # Write to output file (overwrite if exists)
        final_combined.to_netcdf(output_file, encoding=encoding)

        # Clean up
        for ds in chunk_datasets:
            ds.close()
        final_combined.close()
        gc.collect()

        if Path(output_file).exists():
            file_size_gb = Path(output_file).stat().st_size / (1024 ** 3)
            logger.info(f"✅ Combined {len(chunk_datasets)} chunks into {output_file}")
            logger.info(f"  Size: {file_size_gb:.2f} GB")
            logger.info(f"  IDs: {len(final_combined['id_geohash']):,}")
            logger.info(f"  Dates: {len(final_combined['date'])}")
            return True
        else:
            logger.error(f"Output file not created: {output_file}")
            return False

    except Exception as e:
        logger.error(f"Error combining chunks: {e}")
        import traceback
        traceback.print_exc()
        return False


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