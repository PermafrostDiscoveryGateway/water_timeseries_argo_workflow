import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
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
import psutil
from water_timeseries.utils.spatial import create_longitude_latitude_grid, filter_gdf_by_bbox
from water_timeseries.dataset import DWDataset
from water_timeseries.breakpoint import NRTBreakpoint
import datetime
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


def check_analysis_date_in_netcdf(netcdf_path: str, analysis_date: str) -> bool:
    """
    Check if the analysis date exists in the NetCDF file.

    Args:
        netcdf_path: Path to the NetCDF file
        analysis_date: Date in YYYY-MM format

    Returns:
        True if the date exists, False otherwise
    """
    try:
        with xr.open_dataset(netcdf_path) as ds:
            if 'date' in ds.dims or 'date' in ds.coords:
                # Get all dates in the dataset
                dates = ds['date'].values
                # Convert to string if they're datetime objects
                if np.issubdtype(dates.dtype, np.datetime64):
                    date_strings = pd.to_datetime(dates).strftime('%Y-%m')
                else:
                    date_strings = pd.to_datetime(dates).strftime('%Y-%m')

                if analysis_date in date_strings:
                    logger.info(f"Analysis date {analysis_date} found in NetCDF file")
                    return True
                else:
                    logger.error(f"Analysis date {analysis_date} NOT found in NetCDF file")
                    logger.info(f"Available dates in file: {sorted(set(date_strings))[:10]}...")
                    return False
            else:
                logger.error("No 'date' dimension or coordinate found in NetCDF file")
                return False
    except Exception as e:
        logger.error(f"Error checking NetCDF file: {e}")
        return False


def merge_netcdf_chunked(ds_historical, combined_ds, output_path, chunk_size=250):
    """
    Merge historical and combined datasets in chunks.
    Collects all chunks first then concatenates and writes to file.
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

        encoding = {var: {'zlib': True, 'complevel': 5} for var in final_merged.data_vars}

        temp_output = output_path.with_suffix('.tmp.nc')

        final_merged.to_netcdf(temp_output, encoding=encoding, mode='w')

        close_and_clean(final_merged, "final_merged")
        for chunk in merged_chunks:
            close_and_clean(chunk, "merged_chunk")

        if temp_output.exists():
            if output_path.exists():
                output_path.unlink()
            temp_output.rename(output_path)
            logger.info(f"Successfully wrote merged file to {output_path}")
            logger.info(f"File size: {output_path.stat().st_size / (1024 ** 3):.2f} GB")
    else:
        logger.error("No chunks were created, cannot merge")

    return output_path


def run_water_timeseries_analysis(REGION: str, ANALYSIS_DATE: str):
    """
    Run the water timeseries analysis for a specific region and analysis date.

    Args:
        REGION: Region name (e.g., "TEST")
        ANALYSIS_DATE: Analysis date in YYYY-MM format (e.g., "2026-05")
    """
    # Set thread limits
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    log_memory_usage("Program start")

    region_boundaries = get_region_boundaries()

    start = datetime.datetime.now()
    logger.debug(f"Current time: {datetime.datetime.now()}")

    # Environment variables should already be loaded from .env file
    REGION_NAME = REGION

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
        sys.exit(1)

    logger.debug(f"Region name is {REGION_NAME}")

    bounding_box_coords = region_boundaries[REGION_NAME]

    logger.debug(f"Bounding box coordinates are {bounding_box_coords}")
    time.sleep(15)

    X_MIN_START = bounding_box_coords['X_MIN_START']
    X_MIN_END = bounding_box_coords['X_MIN_END']
    Y_MIN_START = bounding_box_coords['Y_MIN_START']
    Y_MIN_END = bounding_box_coords['Y_MIN_END']

    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)

    # Check if the analysis date exists in the NetCDF file
    if not check_analysis_date_in_netcdf(most_recent_dynamic_world_file, ANALYSIS_DATE):
        logger.error(f"Cannot proceed: Analysis date {ANALYSIS_DATE} not found in {most_recent_dynamic_world_file}")
        sys.exit(1)

    hist_file_size_gb = get_file_size_gb(most_recent_dynamic_world_file)
    logger.info(f"Historical NetCDF file size: {hist_file_size_gb:.2f} GB")

    vector_lake_file = os.environ['vector_lake_file']
    path_historical_dw = most_recent_dynamic_world_file
    path_lake_vector = vector_lake_file

    gdf = gpd.read_parquet(path_lake_vector)
    log_memory_usage("After loading lake vectors")

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

    breaks_list = []
    total = len(grid[:])

    # First, load historical dataset once to get valid IDs and also get the data for analysis date
    logger.info("Loading historical dataset to check valid IDs and extract analysis date data...")
    ds_historical_full = xr.open_dataset(path_historical_dw)
    valid_historical_ids = set(ds_historical_full['id_geohash'].values)

    # Extract data for the specific analysis date from the historical dataset
    # Assuming 'date' is a dimension in the dataset
    try:
        # Get the slice for the analysis date
        ds_analysis_date = ds_historical_full.sel(date=ANALYSIS_DATE)
        logger.info(f"Successfully extracted data for date {ANALYSIS_DATE}")
    except (KeyError, ValueError) as e:
        logger.error(f"Could not extract data for date {ANALYSIS_DATE}: {e}")
        ds_historical_full.close()
        sys.exit(1)

    logger.info(f"Found {len(valid_historical_ids)} valid IDs in historical dataset")

    # run loop - now using the pre-extracted analysis date data
    logger.debug(f"There are total {total} grid tiles for {REGION_NAME}")
    time.sleep(15)
    for i, (lon, lat) in enumerate(tqdm(grid[:], total=total, desc="Processing")):
        logger.debug(f"Processing {i}/{total} grid tiles.")
        bbox_west = int(lon)
        bbox_east = int(lon + bbox_size_lon)
        bbox_south = int(lat)
        bbox_north = int(lat + bbox_size_lat)

        print(f"Run processing for bbox: {bbox_west} {bbox_east} {bbox_south} {bbox_north}")

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

        # Subset the analysis date data for these IDs
        ds_analysis_subset = ds_analysis_date.sel(id_geohash=id_list)

        # Subset historical data for these IDs (full time series)
        ds_historical_subset = ds_historical_full.sel(id_geohash=id_list)

        # Merge and process - now using ds_analysis_subset instead of downloaded data
        # Note: ds_analysis_subset is just the slice for the analysis date
        # We need to merge it with the historical data properly
        # Since ds_analysis_subset might have different dimensions, we'll need to align them

        # Option 1: If ds_analysis_subset is just a slice, we can add it as a new variable
        # or expand it to match dimensions. Here's a simple approach:
        ds_merged = ds_historical_subset.copy()

        # Add the analysis date data as a new data variable or replace the slice
        # This depends on your exact data structure. Adjust as needed:
        for var in ds_analysis_subset.data_vars:
            if var in ds_merged.data_vars:
                # If the variable exists, we might want to keep the full time series
                # and just note that this is the analysis date
                logger.debug(f"Variable {var} already exists in historical data")
            else:
                # Add new variable from analysis date
                ds_merged[var] = ds_analysis_subset[var]

        # Sort by date if needed
        if 'date' in ds_merged.dims:
            ds_merged = ds_merged.sortby('date')

        dwds = DWDataset(ds_merged)

        breaks = bp.calculate_break(dataset=dwds, analysis_date=ANALYSIS_DATE)
        breaks.to_parquet(outfile_breaks)
        breaks_list.append(breaks)

        # Clean up
        close_and_clean(ds_historical_subset, f"historical_subset_{i}")
        close_and_clean(ds_analysis_subset, f"analysis_subset_{i}")
        close_and_clean(ds_merged, f"merged_{i}")
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

    # Close the full historical dataset
    close_and_clean(ds_historical_full, "ds_historical_full")

    # Clean up ds_analysis_date reference (already closed via ds_historical_full)

    end = datetime.datetime.now()
    logger.debug(f"Finished processing in {end - start}")

    logger.info("Analysis completed successfully")


def main():
    """Main function that loads .env and calls the analysis function"""
    # Load environment variables
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    # Call the analysis function with sample parameters
    REGION = "TEST"
    ANALYSIS_DATE = "2026-05"

    run_water_timeseries_analysis(REGION, ANALYSIS_DATE)


if __name__ == '__main__':
    main()