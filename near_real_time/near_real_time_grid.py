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
from dotenv import load_dotenv
import time
from loguru import logger
import geemap
import ee
import glob
import os
import gc
import shutil
import psutil
from water_timeseries.utils.spatial import create_longitude_latitude_grid, filter_gdf_by_bbox
from water_timeseries.utils import io
from water_timeseries.dataset import DWDataset
from water_timeseries.breakpoint import NRTBreakpoint
import datetime
import argparse

from utils.region_boundaries import get_region_boundaries
import json
import resource


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


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run breakpoint analysis for a specific date and region (no download)')
    parser.add_argument('analysis_date', type=str, help='Analysis date in YYYY-MM format (e.g., 2024-01)')
    parser.add_argument('--env', type=str, help='Path to .env file', default=None)
    parser.add_argument('--region', type=str, help='Region name (overrides .env file)', default=None)

    args = parser.parse_args()

    # Validate date format
    try:
        analysis_date_obj = datetime.datetime.strptime(args.analysis_date, "%Y-%m")
        ANALYSIS_DATE = args.analysis_date
    except ValueError:
        logger.error(f"Invalid date format: {args.analysis_date}. Please use YYYY-MM format")
        sys.exit(1)

    log_memory_usage("Program start")

    region_boundaries = get_region_boundaries()

    start = datetime.datetime.now()
    logger.debug(f"Current time: {datetime.datetime.now()}")
    logger.info(f"Processing analysis date: {ANALYSIS_DATE}")

    # Load environment variables
    if args.env:
        load_dotenv(dotenv_path=args.env)
        logger.info(f"Loading environment from: {args.env}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    # Set region name (command line argument takes precedence)
    if args.region:
        REGION_NAME = args.region
    else:
        REGION_NAME = os.getenv("region_name", "TEST")

    logger.info(f"Processing region: {REGION_NAME}")

    output_dir = os.environ['output_dir']
    output_dir = os.path.join(output_dir, REGION_NAME)
    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID

    # Initialize Earth Engine (optional - may not be needed if no download)
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
        sys.exit(1)

    logger.debug(f"Region name is {REGION_NAME}")

    bounding_box_coords = region_boundaries[REGION_NAME]

    logger.debug(f"Bounding box coordinates are {bounding_box_coords}")
    time.sleep(5)

    X_MIN_START = bounding_box_coords['X_MIN_START']
    X_MIN_END = bounding_box_coords['X_MIN_END']
    Y_MIN_START = bounding_box_coords['Y_MIN_START']
    Y_MIN_END = bounding_box_coords['Y_MIN_END']

    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)

    hist_file_size_gb = get_file_size_gb(most_recent_dynamic_world_file)
    logger.info(f"Historical NetCDF file size: {hist_file_size_gb:.2f} GB")

    vector_lake_file = os.environ['vector_lake_file']
    path_historical_dw = most_recent_dynamic_world_file
    path_lake_vector = vector_lake_file

    # Load lake vectors
    gdf = gpd.read_parquet(path_lake_vector)
    log_memory_usage("After loading lake vectors")

    bbox_size_lon = 1
    bbox_size_lat = 1
    grid = create_longitude_latitude_grid(lon_range=(X_MIN_START, X_MIN_END),
                                          lat_range=(Y_MIN_START, Y_MIN_END),
                                          bbox_size_lon=bbox_size_lon,
                                          bbox_size_lat=bbox_size_lat)
    logger.info('Created grid')
    log_memory_usage("After creating grid")

    bp = NRTBreakpoint()

    current_breakpoint_dir = Path(output_dir) / f'breakpoint_{ANALYSIS_DATE}'
    current_breakpoint_dir.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Current breakpoint directory: {current_breakpoint_dir}")

    # No download directory needed since we're not downloading
    logger.info("Running in NO-DOWNLOAD mode - using only existing historical data")

    breaks_list = []
    total = len(grid[:])

    # Load historical dataset once to get valid IDs
    logger.info("Loading historical dataset to check valid IDs...")
    ds_historical_check = xr.open_dataset(path_historical_dw)
    valid_historical_ids = set(ds_historical_check['id_geohash'].values)
    ds_historical_check.close()
    logger.info(f"Found {len(valid_historical_ids)} valid IDs in historical dataset")

    # Check if the analysis date exists in the historical dataset
    logger.info(f"Checking if {ANALYSIS_DATE} exists in historical data...")
    ds_date_check = xr.open_dataset(path_historical_dw)
    available_dates = pd.to_datetime(ds_date_check['date'].values)
    ds_date_check.close()

    date_exists = pd.Timestamp(ANALYSIS_DATE) in available_dates
    if not date_exists:
        logger.warning(f"Analysis date {ANALYSIS_DATE} not found in historical dataset!")
        logger.warning(f"Available dates range: {available_dates.min()} to {available_dates.max()}")
        logger.warning(f"Will still attempt to run breakpoint analysis, but may fail if date is missing")

    logger.info(f"Processing {total} grid tiles for {REGION_NAME}")

    for i, (lon, lat) in enumerate(tqdm(grid[:], total=total, desc="Processing")):
        logger.debug(f"Processing {i}/{total} grid tiles.")
        bbox_west = int(lon)
        bbox_east = int(lon + bbox_size_lon)
        bbox_south = int(lat)
        bbox_north = int(lat + bbox_size_lat)

        logger.debug(f"Run processing for bbox: {bbox_west} {bbox_east} {bbox_south} {bbox_north}")

        outfile_breaks = current_breakpoint_dir / f'DW_{ANALYSIS_DATE}_{bbox_west}_{bbox_east}_{bbox_south}_{bbox_north}_breaks.parquet'

        if outfile_breaks.exists():
            logger.debug(f'Breakpoints already calculated! Skipping {bbox_west} {bbox_south}')
            breaks_list.append(pd.read_parquet(outfile_breaks))
            continue

        gdf_subset = filter_gdf_by_bbox(gdf=gdf,
                                        bbox_west=lon,
                                        bbox_east=lon + bbox_size_lon,
                                        bbox_south=lat,
                                        bbox_north=lat + bbox_size_lat)
        n_lakes = len(gdf_subset)
        logger.debug(f'Number of lakes: {n_lakes}')

        id_list = gdf_subset['id_geohash'].values.tolist()
        if n_lakes == 0:
            logger.debug(f'No lakes for grid {bbox_west} {bbox_south}. Skipping!')
            continue

        # Filter IDs to only those that exist in historical data
        original_count = len(id_list)
        id_list = [id_val for id_val in id_list if id_val in valid_historical_ids]
        filtered_count = len(id_list)

        if filtered_count == 0:
            logger.warning(
                f'No valid historical IDs for grid {bbox_west} {bbox_south} (had {original_count} lakes, none in historical data). Skipping!')
            continue
        elif filtered_count < original_count:
            logger.info(
                f'Filtered {original_count - filtered_count} lakes not found in historical data. Processing {filtered_count} lakes.')
            # Also filter the gdf_subset to only keep valid IDs
            gdf_subset = gdf_subset[gdf_subset['id_geohash'].isin(id_list)]

        # Load historical data for this tile
        logger.debug(f"Loading historical dataset for tile {i}...")
        ds_historical = xr.open_dataset(path_historical_dw)

        # Subset historical data
        ds_historical_subset = ds_historical.sel(id_geohash=id_list)

        # Close historical immediately
        ds_historical.close()
        del ds_historical
        gc.collect()

        # Use only historical data (no download/merge needed)
        ds_merged = ds_historical_subset

        # Create dataset and calculate breakpoints
        dwds = DWDataset(ds_merged)
        breaks = bp.calculate_break(dataset=dwds, analysis_date=ANALYSIS_DATE)
        breaks.to_parquet(outfile_breaks)
        breaks_list.append(breaks)

        # Clean up
        ds_historical_subset.close()
        ds_merged.close()
        del ds_historical_subset, ds_merged
        gc.collect()

        # Periodic save
        if len(breaks_list) >= 10:
            logger.info(f"Saving intermediate results...")
            breaks_merged = pd.concat(breaks_list, ignore_index=True)
            joined = gdf.set_index('id_geohash').join(breaks_merged, how='inner').reset_index()
            partial_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}_partial.parquet'
            joined.to_parquet(partial_file)
            breaks_list = []
            gc.collect()

    # Final save
    if breaks_list:
        breaks_merged = pd.concat(breaks_list, ignore_index=True)
        joined = gdf.set_index('id_geohash').join(breaks_merged, how='inner').reset_index()
        path_to_joined_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}.parquet'
        joined.to_parquet(path_to_joined_file)
        logger.info(f"Final combined file saved to {path_to_joined_file}")

    end = datetime.datetime.now()
    logger.info(f"Finished processing in {end - start}")

    # Note: Zarr file creation is skipped since no new data was downloaded
    logger.info("No Zarr file created because no new data was downloaded")
    logger.info("Script completed successfully")


if __name__ == '__main__':
    main()