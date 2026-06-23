import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
import sys
from pathlib import Path

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


def create_merged_netcdf_memory_efficient(ds_historical, combined_ds, output_path, chunk_size=500):
    """
    Create a merged NetCDF file in a memory-efficient way by processing chunks sequentially.
    Uses a safer approach with temporary files and xarray's combine functionality.

    Args:
        ds_historical: Historical dataset
        combined_ds: Combined (new) dataset
        output_path: Path for output NetCDF file
        chunk_size: Number of IDs to process at once
    """
    logger.info(f"Creating merged NetCDF file at {output_path}")
    log_memory_usage("Start of merge_netcdf")

    # Get all unique IDs - ensure they are sorted
    hist_ids = set(ds_historical['id_geohash'].values)
    combined_ids = set(combined_ds['id_geohash'].values)
    all_ids = np.array(sorted(hist_ids | combined_ids))  # Ensure sorted

    logger.info(f"Historical IDs: {len(hist_ids)}, Combined IDs: {len(combined_ids)}")
    logger.info(f"Total unique IDs: {len(all_ids)}")

    total_ids = len(all_ids)
    num_chunks = (total_ids + chunk_size - 1) // chunk_size
    logger.info(f"Processing {total_ids} IDs in {num_chunks} chunks of {chunk_size}")

    # Store paths to chunk files for later combination
    chunk_files = []
    temp_dir = output_path.parent / f"temp_chunks_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    temp_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Using temporary directory: {temp_dir}")

    try:
        for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_ids)
            chunk_ids = all_ids[start_idx:end_idx]

            logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks}: IDs {start_idx} to {end_idx}")
            log_memory_usage(f"Chunk {chunk_idx + 1} start")

            try:
                # Convert to list for selection
                chunk_ids_list = chunk_ids.tolist()

                # Get data for this chunk from historical dataset
                hist_chunk = None
                try:
                    # Filter to only IDs that exist
                    existing_hist_ids = [id_val for id_val in chunk_ids_list if id_val in hist_ids]
                    if existing_hist_ids:
                        hist_chunk = ds_historical.sel(id_geohash=existing_hist_ids)
                except Exception as e:
                    logger.warning(f"Error selecting from historical: {e}")

                # Get data for this chunk from combined dataset
                combined_chunk = None
                try:
                    existing_combined_ids = [id_val for id_val in chunk_ids_list if id_val in combined_ids]
                    if existing_combined_ids:
                        combined_chunk = combined_ds.sel(id_geohash=existing_combined_ids)
                except Exception as e:
                    logger.warning(f"Error selecting from combined: {e}")

                # Merge the chunks
                if hist_chunk is not None and combined_chunk is not None:
                    # Both have data - merge them
                    merged_chunk = xr.concat([hist_chunk, combined_chunk], dim='id_geohash')
                    # Remove duplicate IDs if any
                    _, unique_indices = np.unique(merged_chunk['id_geohash'].values, return_index=True)
                    if len(unique_indices) < len(merged_chunk['id_geohash']):
                        merged_chunk = merged_chunk.isel(id_geohash=np.sort(unique_indices))
                elif hist_chunk is not None:
                    # Only historical data
                    merged_chunk = hist_chunk
                elif combined_chunk is not None:
                    # Only combined data
                    merged_chunk = combined_chunk
                else:
                    logger.warning(f"No data found for chunk {chunk_idx + 1}, skipping")
                    continue

                # Sort for consistency
                merged_chunk = merged_chunk.sortby(['date', 'id_geohash'])

                # Write this chunk to a temporary file
                chunk_file = temp_dir / f"chunk_{chunk_idx:06d}.nc"

                # Use compression but simpler encoding
                encoding = {}
                for var in merged_chunk.data_vars:
                    encoding[var] = {
                        'zlib': True,
                        'complevel': 2,
                        'shuffle': True
                    }

                # Write the chunk file
                merged_chunk.to_netcdf(chunk_file, encoding=encoding)
                chunk_files.append(chunk_file)

                logger.info(f"Chunk {chunk_idx + 1} written to {chunk_file} ({len(merged_chunk['id_geohash'])} IDs)")

                # Clean up
                if hist_chunk is not None:
                    hist_chunk.close()
                if combined_chunk is not None:
                    combined_chunk.close()
                merged_chunk.close()
                del merged_chunk
                if hist_chunk is not None:
                    del hist_chunk
                if combined_chunk is not None:
                    del combined_chunk
                gc.collect()

                log_memory_usage(f"Chunk {chunk_idx + 1} complete")

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx + 1}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        # Combine all chunk files into one
        if chunk_files:
            logger.info(f"Combining {len(chunk_files)} chunk files into final NetCDF...")
            log_memory_usage("Before combining chunks")

            # Open all chunk files and combine
            combined_chunks = []
            for chunk_file in tqdm(chunk_files, desc="Opening chunk files"):
                try:
                    ds = xr.open_dataset(chunk_file)
                    combined_chunks.append(ds)
                except Exception as e:
                    logger.error(f"Error opening chunk file {chunk_file}: {e}")

            if combined_chunks:
                # Concatenate all chunks
                logger.info("Concatenating chunks...")
                final_ds = xr.concat(combined_chunks, dim='id_geohash')

                # Remove duplicates (just in case)
                _, unique_indices = np.unique(final_ds['id_geohash'].values, return_index=True)
                if len(unique_indices) < len(final_ds['id_geohash']):
                    final_ds = final_ds.isel(id_geohash=np.sort(unique_indices))

                # Sort by date and id
                final_ds = final_ds.sortby(['date', 'id_geohash'])

                logger.info(f"Final dataset has {len(final_ds['id_geohash'])} IDs")
                logger.info(f"Date range: {final_ds['date'].min().values} to {final_ds['date'].max().values}")

                # Write final file
                logger.info(f"Writing final NetCDF file to {output_path}")

                # Use compression
                encoding = {}
                for var in final_ds.data_vars:
                    encoding[var] = {
                        'zlib': True,
                        'complevel': 4,
                        'shuffle': True
                    }

                # Write in one go (dataset should be manageable now)
                final_ds.to_netcdf(output_path, encoding=encoding)

                # Verify
                if output_path.exists():
                    file_size_gb = get_file_size_gb(str(output_path))
                    logger.info(f"Successfully created merged NetCDF file: {output_path}")
                    logger.info(f"File size: {file_size_gb:.2f} GB")

                    # Verify readability
                    test_ds = xr.open_dataset(output_path)
                    logger.info(f"Verification: File has {len(test_ds['id_geohash'])} IDs")
                    test_ds.close()

                # Clean up
                final_ds.close()
                del final_ds
                for ds in combined_chunks:
                    ds.close()
                del combined_chunks
                gc.collect()
            else:
                logger.error("No chunks could be opened")
        else:
            logger.error("No chunk files were created")

    finally:
        # Clean up temporary directory
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
            logger.info("Temporary directory cleaned up")
        except Exception as e:
            logger.warning(f"Could not clean up temporary directory: {e}")

    log_memory_usage("End of merge_netcdf")
    return output_path


def near_real_time_region(region: str = "TEST", env_path: str = None):
    """
    Run near-real-time breakpoint analysis for a specific region.

    Args:
        region: Region name (e.g., "TEST", "AFRICA", "SOUTH_AMERICA")
               Defaults to "TEST"
        env_path: Optional path to .env file. If None, uses default .env
    """
    # Set thread limits
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # os.environ['NUMEXPR_NUM_THREADS'] = '1'

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
        most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
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
                    try:
                        ds_dl = downloader.download_dw_monthly(
                            gdf=gdf_subset,
                            max_total_requests=2000,
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
                    chunk_size=1000  # Adjust based on available memory
                )

                logger.info(f"Original file: {most_recent_dynamic_world_file}")
                logger.info(f"New file: {new_historical_path}")

                # Clean up
                combined.close()
                del combined
                gc.collect()
            else:
                logger.warning("No combined data to merge")

        # Continue with existing merge_zarr_chunked logic
        if combined is not None and ds_historical is not None:
            merge_zarr_chunked(ds_historical, combined, output_zarr, chunk_size=250)
            ds_historical.close()

        logger.debug(f"End of date block for {REGION_NAME} and date {date}")

    logger.info(f"Near-real-time processing completed for region: {REGION_NAME}")
    return True


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

    success = near_real_time_region(region=args.region, env_path=args.env_path)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()