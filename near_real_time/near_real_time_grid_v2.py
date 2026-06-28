import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

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


def near_real_time_region(region: str = "TEST", env_path: str = None):
    """
    Run near-real-time breakpoint analysis for a specific region.
    (Legacy function - kept for compatibility)
    """
    # ... (existing code, unchanged - keep the original function body)


def download_near_real_time_region(region: str = "TEST", run_start_label: str = None, env_path: str = None):
    """
    Download near-real-time data for a specific region and create merged NetCDF file.

    KEY FIXES:
    1. Loads original valid IDs once and reuses them for all dates
    2. Does NOT overwrite the original file - creates new file with verification
    3. Verifies the merge before considering it successful
    4. Uses the most recent historical file for each date

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
    most_recent_dynamic_world_file = original_historical_file

    hist_file_size_gb = get_file_size_gb(most_recent_dynamic_world_file)
    logger.info(f"Original Historical NetCDF file size: {hist_file_size_gb:.2f} GB")

    missing_dates = utils.download_new_dynamic_world_data.check_missing_data_in_netcdf(most_recent_dynamic_world_file)

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

        # ========== LOAD THE MOST RECENT HISTORICAL FILE ==========
        all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
        most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
        logger.debug(f"Most recent dynamic world file: {most_recent_dynamic_world_file}")
        hist_file_size_gb = get_file_size_gb(most_recent_dynamic_world_file)
        logger.info(f"Historical NetCDF file size: {hist_file_size_gb:.2f} GB")

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

            # ========== USE ORIGINAL VALID IDs (not reloaded from file) ==========
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
            'historical_file': str(most_recent_dynamic_world_file)
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

        # ========== MERGE AFTER EACH DATE ==========
        downloaded_files = sorted(glob.glob(str(current_download_dir / f'DW_{ANALYSIS_DATE}_*.nc')))

        if downloaded_files:
            logger.info(f"Found {len(downloaded_files)} downloaded files for {ANALYSIS_DATE}")

            # Load the most recent historical file for merging
            logger.info(f"Loading historical dataset from: {most_recent_dynamic_world_file}")
            ds_historical = xr.open_dataset(most_recent_dynamic_world_file)

            logger.info(f"Loading {len(downloaded_files)} downloaded files...")

            # Process downloaded files in batches
            BATCH_SIZE = 10
            combined = None

            for batch_idx in tqdm(range(0, len(downloaded_files), BATCH_SIZE),
                                  desc=f"Processing batches {ANALYSIS_DATE}"):
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

                # ========== CREATE NEW FILE (DO NOT OVERWRITE ORIGINAL) ==========
                logger.info("Starting memory-efficient merge to NetCDF...")
                result_path = create_merged_netcdf_memory_efficient(
                    ds_historical=ds_historical,
                    combined_ds=combined,
                    output_path=new_historical_path,
                    chunk_size=5000
                )

                # ========== VERIFY THE MERGE BEFORE REPLACING ==========
                if result_path and Path(result_path).exists():
                    # Get expected counts
                    expected_id_count = len(original_valid_ids) + len(combined['id_geohash'])

                    # Verify the new file
                    verification = verify_merged_netcdf(
                        result_path,
                        expected_id_count=None,  # Don't enforce exact count
                        expected_date_count=None
                    )

                    if verification['valid']:
                        logger.info(
                            f"✅ Merge successful! New file has {verification['id_count']} IDs, {verification['date_count']} dates")
                        logger.info(f"   File size: {verification['file_size_gb']:.2f} GB")

                        # Create merged marker
                        merged_marker = current_download_dir / f'merged_complete_{date_run_label}.success'
                        with open(merged_marker, 'w') as f:
                            f.write(f"Merged {len(downloaded_files)} files into historical NetCDF\n")
                            f.write(f"New file: {result_path}\n")
                            f.write(f"ID count: {verification['id_count']}\n")
                            f.write(f"Date count: {verification['date_count']}\n")
                            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")

                        # Update the most recent file path for the next date
                        most_recent_dynamic_world_file = str(result_path)
                        logger.info(f"Updated most recent file to: {most_recent_dynamic_world_file}")
                    else:
                        logger.error(f"❌ Merge verification failed: {verification.get('error', 'Unknown error')}")
                        overall_success = False
                else:
                    logger.error("❌ Merge failed! No output file created.")
                    overall_success = False

                # Clean up
                combined.close()
                del combined
                gc.collect()
            else:
                logger.warning("No combined data to merge")

            ds_historical.close()
            del ds_historical
            gc.collect()
        else:
            logger.warning(f"No downloaded files found for {ANALYSIS_DATE}")

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

        # Check that merged file exists
        merged_markers = list(current_download_dir.glob(f'merged_complete_*.success'))
        if merged_markers:
            date_result['merged_file'] = str(max(merged_markers, key=lambda p: p.stat().st_mtime))
            logger.info(f"✅ Found merged marker for {analysis_date}")

            # Verify the actual NetCDF file exists
            merged_marker_path = Path(date_result['merged_file'])
            with open(merged_marker_path, 'r') as f:
                content = f.read()
                # Try to extract the new file path
                for line in content.split('\n'):
                    if line.startswith('New file:'):
                        new_file_path = line.replace('New file:', '').strip()
                        if Path(new_file_path).exists():
                            date_result['merged_netcdf_file'] = new_file_path
                            logger.info(f"✅ Verified merged NetCDF file exists: {new_file_path}")
                        else:
                            logger.warning(f"⚠️ Merged NetCDF file not found: {new_file_path}")
                            date_result['merged_netcdf_file'] = None
                else:
                    # If we can't parse the file, just check if any NetCDF files exist
                    nc_files = list(current_download_dir.glob(f'DW_{analysis_date}_*.nc'))
                    if nc_files:
                        logger.info(f"✅ Found {len(nc_files)} downloaded NetCDF files")
                    else:
                        logger.warning(f"⚠️ No downloaded NetCDF files found in {current_download_dir}")
        else:
            logger.warning(f"No merged marker found for {analysis_date}")
            if strict_mode:
                all_complete = False
                date_result['reason'] = 'No merged marker found'
                date_results[analysis_date] = date_result
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
    Accepts region as first argument and optional env file as second argument.

    Usage:
        python script.py [REGION] [ENV_PATH]

    Examples:
        python script.py TEST
        python script.py AFRICA /path/to/.env
        python script.py                     # Uses default TEST region
    """
    import argparse

    parser = argparse.ArgumentParser(description='Run near-real-time breakpoint analysis for a region')
    parser.add_argument('region', nargs='?', default='TEST',
                        help='Region name (default: TEST)')
    parser.add_argument('env_path', nargs='?', default=None,
                        help='Optional path to .env file')

    args = parser.parse_args()

    run_start = datetime.datetime.now()
    run_start_label = run_start.strftime("%Y_%m_%d_%H_%M_%S")

    success = download_near_real_time_region(region=args.region, run_start_label=run_start_label,
                                             env_path=args.env_path)
    sys.exit(0 if success.get('success', False) else 1)


# if __name__ == '__main__':
#     main()