import geopandas as gpd
# !/usr/bin/env python3
import sys
from pathlib import Path
from pathlib import Path

# Add the parent directory to Python path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# def setup_water_timeseries_path():
#     """
#     Dynamically add water-timeseries-v2 to Python path.
#     Works from any location without hardcoded paths.
#     """
#     # Get the current script's location
#     script_path = Path(__file__).resolve()
#
#     # Look for water-timeseries-v2 by traversing up from script location
#     # and checking common sibling/parent locations
#     candidates = []
#
#     # Add current script's directory and parents
#     for parent in [script_path.parent] + list(script_path.parents):
#         candidates.append(parent / "water-timeseries-v2")
#         candidates.append(parent.parent / "water-timeseries-v2")
#         candidates.append(parent / "../water-timeseries-v2")
#
#     # Add common locations relative to home
#     candidates.append(Path.home() / "water-timeseries-v2")
#
#     # Check if the package is already importable
#     try:
#         import water_timeseries
#         print(f"✓ water_timeseries already available at: {water_timeseries.__file__}")
#         return True
#     except ImportError:
#         pass
#
#     # Try each candidate
#     for candidate in candidates:
#         src_path = candidate.resolve() / "src"
#         if src_path.exists() and (src_path / "water_timeseries").exists():
#             print(f"✓ Found water_timeseries at: {src_path}")
#             sys.path.insert(0, str(src_path))
#             return True
#
#     print("⚠️  WARNING: Could not find water-timeseries-v2")
#     print("   Searched in:")
#     for c in candidates[:5]:
#         print(f"     - {c}")
#     return False


# Run the setup
# if not setup_water_timeseries_path():
#     print("ERROR: water_timeseries package not found. Please check installation.")
#     sys.exit(1)

import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add project root to Python path
# PROJECT_ROOT = Path("/home/ext_tcnichol_illinois_edu/water-timeseries-v2")
# SRC_PATH = PROJECT_ROOT / "src"
#
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))
# if str(SRC_PATH) not in sys.path:
#     sys.path.insert(0, str(SRC_PATH))
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
from water_timeseries.breakpoint import BeastBreakpoint  # Changed from NRTBreakpoint
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
    """
    log_memory_usage("Program start")

    region_boundaries = get_region_boundaries()

    start = datetime.datetime.now()
    logger.debug(f"Current time: {datetime.datetime.now()}")

    # Environment variables should already be loaded from .env file
    REGION_NAME = REGION

    output_dir = os.environ['output_dir']
    output_dir = os.path.join(output_dir, REGION_NAME, 'historical')
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
    logger.info(f"Total lakes in vector file: {len(gdf)}")

    bbox_size_lon = 1
    bbox_size_lat = 1
    grid = create_longitude_latitude_grid(lon_range=(X_MIN_START, X_MIN_END), lat_range=(Y_MIN_START, Y_MIN_END),
                                          bbox_size_lon=bbox_size_lon, bbox_size_lat=bbox_size_lat)
    print('created grid')
    log_memory_usage("After creating grid")

    # Use BeastBreakpoint with default threshold of 0.5
    bp = BeastBreakpoint()

    current_breakpoint_dir = Path(output_dir) / f'breakpoint_{ANALYSIS_DATE}'
    current_breakpoint_dir.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Current breakpoint directory: {current_breakpoint_dir}")

    breaks_list = []
    total = len(grid[:])

    # Load historical dataset once
    logger.info("Loading historical dataset...")
    ds_historical_full = xr.open_dataset(path_historical_dw)
    valid_historical_ids = set(ds_historical_full['id_geohash'].values)
    logger.info(f"Found {len(valid_historical_ids)} valid IDs in historical dataset")

    logger.info(f"Historical dataset includes date {ANALYSIS_DATE} and all previous dates")
    logger.debug(f"There are total {total} grid tiles for {REGION_NAME}")

    # Track statistics
    total_breaks_found = 0
    lakes_with_breaks = 0

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
            df_existing = pd.read_parquet(outfile_breaks)
            breaks_list.append(df_existing)
            total_breaks_found += len(df_existing)
            lakes_with_breaks += df_existing['id_geohash'].nunique() if 'id_geohash' in df_existing.columns else 0
            continue

        gdf_subset = filter_gdf_by_bbox(gdf=gdf, bbox_west=lon, bbox_east=lon + bbox_size_lon, bbox_south=lat,
                                        bbox_north=lat + bbox_size_lat)
        n_lakes = len(gdf_subset)
        print('Number of lakes: ', n_lakes)

        id_list = gdf_subset['id_geohash'].values.tolist()
        if n_lakes == 0:
            print(f'No lakes for grid {bbox_west} {bbox_south}. Skipping!')
            continue
        else:
            logger.debug(f"We have lakes to process {n_lakes} grids")

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

        # Subset historical data for these IDs
        ds_historical_subset = ds_historical_full.sel(id_geohash=id_list)

        # Reorder dimensions
        try:
            ds_historical_subset = ds_historical_subset.transpose('date', 'id_geohash')
            logger.debug(f"Reordered dimensions to: {list(ds_historical_subset.dims)}")
        except ValueError as e:
            logger.warning(f"Could not reorder dimensions: {e}")

        # Create DWDataset
        dwds = DWDataset(ds_historical_subset)

        logger.debug(f"Dataset dims after reorder: {ds_historical_subset.dims}")
        logger.debug(f"Water DataFrame index: {ds_historical_subset[dwds.water_column].to_dataframe().index.names}")

        # Calculate breakpoints for each lake
        for object_id in id_list:
            try:
                break_result = bp.calculate_break(dataset=dwds, object_id=object_id)

                if not break_result.empty:
                    # Ensure id_geohash is a column, not just an index
                    if 'id_geohash' not in break_result.columns:
                        # Reset index to make id_geohash a column
                        break_result = break_result.reset_index()
                        logger.debug(f"Reset index for lake {object_id}, now columns: {break_result.columns.tolist()}")

                    # Verify id_geohash column exists and has correct value
                    if 'id_geohash' in break_result.columns:
                        # Ensure the id_geohash value is correct
                        if break_result['id_geohash'].iloc[0] != object_id:
                            logger.warning(
                                f"ID mismatch: expected {object_id}, got {break_result['id_geohash'].iloc[0]}")
                            break_result['id_geohash'] = object_id

                        breaks_list.append(break_result)
                        total_breaks_found += len(break_result)
                        lakes_with_breaks += 1
                        logger.debug(f"Found {len(break_result)} break(s) for lake {object_id}")
                    else:
                        logger.error(f"id_geohash column still missing after reset_index for lake {object_id}")
                        logger.error(f"Available columns: {break_result.columns.tolist()}")

            except Exception as e:
                logger.error(f"Error calculating break for lake {object_id}: {e}")
                continue

        # Save intermediate results periodically
        if len(breaks_list) >= 10:
            logger.info(f"Saving intermediate results with {len(breaks_list)} break DataFrames...")
            if breaks_list:
                breaks_merged = pd.concat(breaks_list, ignore_index=True)
                logger.info(f"Merged {len(breaks_merged)} break records")

                # Log column info for debugging
                logger.debug(f"breaks_merged columns: {breaks_merged.columns.tolist()}")
                logger.debug(f"breaks_merged index: {breaks_merged.index.name}")

                # Check if id_geohash column exists
                if 'id_geohash' not in breaks_merged.columns:
                    logger.error("id_geohash column missing from breaks_merged!")
                    logger.error(f"Available columns: {breaks_merged.columns.tolist()}")
                    # Try to recover by resetting index if id_geohash is the index
                    if breaks_merged.index.name == 'id_geohash':
                        breaks_merged = breaks_merged.reset_index()
                        logger.info("Recovered by resetting index")

                # Now join with gdf
                if 'id_geohash' in breaks_merged.columns:
                    joined = gdf.set_index('id_geohash').join(
                        breaks_merged.set_index('id_geohash'),
                        how='inner'
                    ).reset_index()

                    partial_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}_partial.parquet'
                    joined.to_parquet(partial_file)
                    logger.info(f"Saved {len(joined)} lakes with breaks to partial file")
                else:
                    logger.error("Cannot save partial file - missing id_geohash column")
                    logger.error(f"breaks_merged head:\n{breaks_merged.head()}")

                breaks_list = []
                gc.collect()

        # Clean up
        close_and_clean(ds_historical_subset, f"historical_subset_{i}")
        gc.collect()

    # Final save
    logger.info(
        f"Processing complete. Total breaks found: {total_breaks_found}, Lakes with breaks: {lakes_with_breaks}")

    if breaks_list:
        logger.info(f"Final save with {len(breaks_list)} break DataFrames...")
        breaks_merged = pd.concat(breaks_list, ignore_index=True)
        logger.info(f"Final merged {len(breaks_merged)} break records")

        # Log column info for debugging
        logger.debug(f"Final breaks_merged columns: {breaks_merged.columns.tolist()}")

        # Ensure id_geohash is a column
        if 'id_geohash' not in breaks_merged.columns:
            logger.warning("id_geohash column missing, attempting to reset index")
            if breaks_merged.index.name == 'id_geohash':
                breaks_merged = breaks_merged.reset_index()
                logger.info("Reset index successfully")
            else:
                logger.error(f"Cannot recover - index name is {breaks_merged.index.name}")
                logger.error(f"breaks_merged head:\n{breaks_merged.head()}")

        # Join with gdf if id_geohash column exists
        if 'id_geohash' in breaks_merged.columns:
            joined = gdf.set_index('id_geohash').join(
                breaks_merged.set_index('id_geohash'),
                how='inner'
            ).reset_index()

            path_to_joined_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}.parquet'
            joined.to_parquet(path_to_joined_file)
            logger.info(
                f"Final combined file saved to {path_to_joined_file} with {len(joined)} lakes and {len(breaks_merged)} total breaks")

            # Also save just the breakpoints without geometry for easier analysis
            breaks_only_file = current_breakpoint_dir / f'breakpoints_only_{ANALYSIS_DATE}.parquet'
            breaks_merged.to_parquet(breaks_only_file)
            logger.info(f"Breakpoints-only file saved to {breaks_only_file}")
        else:
            logger.error("Cannot save final file - missing id_geohash column")
            # Save raw breaks for debugging
            debug_file = current_breakpoint_dir / f'debug_breaks_{ANALYSIS_DATE}.parquet'
            breaks_merged.to_parquet(debug_file)
            logger.info(f"Saved raw breaks to {debug_file} for debugging")
    else:
        logger.warning("No breakpoints found for any lakes!")

    # Close the full historical dataset
    close_and_clean(ds_historical_full, "ds_historical_full")

    end = datetime.datetime.now()
    logger.info(f"Finished processing in {end - start}")
    logger.info("Analysis completed successfully")