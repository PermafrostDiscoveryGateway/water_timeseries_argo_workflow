import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import sys
import geemap
import ee
import glob
import os
import gc
import psutil
from water_timeseries.downloader import EarthEngineDownloader
from water_timeseries.utils.spatial import create_longitude_latitude_grid, filter_gdf_by_bbox
from water_timeseries.dataset import DWDataset
from water_timeseries.breakpoint import NRTBreakpoint
import datetime
from region_boundaries import get_region_boundaries
import download_new_dynamic_world_data
import shutil
import json
import resource


def log_memory_usage(stage: str):
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    mem_gb = mem_mb / 1024

    # Get additional memory info
    try:
        # RSS (Resident Set Size) in GB
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        logger.debug(f"[MEMORY] {stage}: {mem_mb:.2f} MB ({mem_gb:.2f} GB) | Max RSS: {rss_gb:.2f} GB")
    except:
        logger.debug(f"[MEMORY] {stage}: {mem_mb:.2f} MB ({mem_gb:.2f} GB)")

    # Warn if memory is high
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
        log_memory_usage(f"After closing {name}")


def merge_netcdf_chunked(ds_historical, combined_ds, output_path, chunk_size=250):
    """
    Merge historical and combined datasets in chunks.
    Collects all chunks first then concatenates and writes to file.
    This avoids the dimension size error when appending.
    """
    logger.info(f"Merging in chunks of {chunk_size} ids")
    log_memory_usage("Before chunked merge")

    # Get all unique ids from combined dataset
    combined_ids = combined_ds['id_geohash'].values
    total_ids = len(combined_ids)
    logger.info(f"Total ids to merge: {total_ids}")

    # List to hold merged chunks
    merged_chunks = []

    # Process in chunks
    for chunk_start in tqdm(range(0, total_ids, chunk_size), desc="Merging chunks"):
        chunk_end = min(chunk_start + chunk_size, total_ids)
        chunk_ids = combined_ids[chunk_start:chunk_end]

        logger.debug(f"Processing chunk: ids {chunk_start} to {chunk_end} ({len(chunk_ids)} ids)")
        log_memory_usage(f"Chunk {chunk_start // chunk_size + 1} start")

        # Subset both datasets for this chunk
        hist_chunk = ds_historical.sel(id_geohash=chunk_ids)
        new_chunk = combined_ds.sel(id_geohash=chunk_ids)

        # Merge just this chunk
        merged_chunk = xr.merge([hist_chunk, new_chunk])
        merged_chunks.append(merged_chunk)

        # Clean up chunk data to free memory
        close_and_clean(hist_chunk, f"hist_chunk_{chunk_start}")
        close_and_clean(new_chunk, f"new_chunk_{chunk_start}")

        log_memory_usage(f"Chunk {chunk_start // chunk_size + 1} complete")

    # Concatenate all chunks and write to file
    logger.info("Concatenating all chunks and writing to file...")
    if merged_chunks:
        final_merged = xr.concat(merged_chunks, dim='id_geohash')

        encoding = {var: {'zlib': True, 'complevel': 5} for var in final_merged.data_vars}

        temp_output = output_path.with_suffix('.tmp.nc')

        # Write the final merged dataset
        final_merged.to_netcdf(temp_output, encoding=encoding, mode='w')

        # Clean up
        close_and_clean(final_merged, "final_merged")
        for chunk in merged_chunks:
            close_and_clean(chunk, "merged_chunk")

        # Rename temp file to final output
        if temp_output.exists():
            if output_path.exists():
                output_path.unlink()
            temp_output.rename(output_path)
            logger.info(f"Successfully wrote merged file to {output_path}")
            logger.info(f"File size: {output_path.stat().st_size / (1024 ** 3):.2f} GB")
    else:
        logger.error("No chunks were created, cannot merge")

    return output_path


def main():
    # Set thread limits to prevent excessive memory usage
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    # Start memory logging
    log_memory_usage("Program start")

    region_boundaries = get_region_boundaries()

    start = datetime.datetime.now()
    logger.debug(f"Current time: {datetime.datetime.now()}")

    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    output_dir = os.environ['output_dir']
    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID

    logger.debug(f"Trying earth engine initialize")
    try:
        ee.Initialize(project=EE_PROJECT_ID)
        logger.debug("Earth engine successfully initialized")
    except Exception as e:
        logger.debug("Failed to initialize earth engine")
        logger.debug(e)

    logger.debug(f"Version of geemap is {geemap.__version__}")

    try:
        geemap.ee_initialize(project=EE_PROJECT_ID)
        logger.debug("Initialized geemap")
    except Exception as e:
        logger.debug("Failed to initialize geemap ")
        logger.debug(e)

    current_region = os.getenv('CURRENT_REGION', 'TEST')

    dynamic_world_data_dir = os.environ['dynamic_world_data']
    dynamic_world_download_dir = Path(os.environ['dynamic_world_downloads'])
    dynamic_world_download_dir.mkdir(exist_ok=True, parents=True)
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))

    if not all_dynamic_world_files:
        logger.error(f"No .nc files found in {dynamic_world_data_dir}")
        sys.exit(1)

    bounding_box_coords = region_boundaries['TEST']

    X_MIN_START = bounding_box_coords['X_MIN_START']
    X_MIN_END = bounding_box_coords['X_MIN_END']
    Y_MIN_START = bounding_box_coords['Y_MIN_START']
    Y_MIN_END = bounding_box_coords['Y_MIN_END']

    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)

    # Log historical file size
    hist_file_size_gb = get_file_size_gb(most_recent_dynamic_world_file)
    logger.info(f"Historical NetCDF file size: {hist_file_size_gb:.2f} GB")

    missing_dates = download_new_dynamic_world_data.check_missing_data_in_netcdf(most_recent_dynamic_world_file, )
    missing_analysis_dates = []
    for date in missing_dates:
        missing_date_string = date.strftime("%Y-%m")
        logger.debug(f"We are missing {missing_date_string}")
        missing_analysis_dates.append(missing_date_string)
    vector_lake_file = os.environ['vector_lake_file']

    # lake vector path
    path_historical_dw = most_recent_dynamic_world_file
    # historical DW data path
    path_lake_vector = vector_lake_file

    ANALYSIS_DATE = "2026-05"

    # read lake vectors
    gdf = gpd.read_parquet(path_lake_vector)
    log_memory_usage("After loading lake vectors")

    bbox_size_lon = 1
    bbox_size_lat = 1
    grid = create_longitude_latitude_grid(lon_range=(X_MIN_START, X_MIN_END), lat_range=(Y_MIN_START, Y_MIN_END),
                                          bbox_size_lon=bbox_size_lon, bbox_size_lat=bbox_size_lat)
    print('created grid')
    log_memory_usage("After creating grid")

    bp = NRTBreakpoint()

    # create directory for current data run
    current_breakpoint_dir = Path(output_dir) / f'breakpoint_{ANALYSIS_DATE}'
    current_breakpoint_dir.mkdir(exist_ok=True, parents=True)

    # create directory for partial dynamic world downloads
    current_download_dir = Path(str(dynamic_world_download_dir), f'download_{ANALYSIS_DATE}')
    current_download_dir.mkdir(exist_ok=True, parents=True)

    if not hasattr(geemap, 'ee_initialize'):
        logger.warning("geemap.ee_initialize missing, adding runtime patch")

        def ee_initialize(project=None, **kwargs):
            if project:
                ee.Initialize(project=project, **kwargs)
            else:
                ee.Initialize(**kwargs)

        geemap.ee_initialize = ee_initialize
        logger.info("Runtime patch applied to geemap")

    # setup downloader
    downloader = EarthEngineDownloader(ee_project=EE_PROJECT_ID)

    breaks_list = []
    total = len(grid[:])

    # Track if we've saved a partial file
    partial_saved = False

    # run loop
    for i, (lon, lat) in enumerate(tqdm(grid[:], total=total, desc="Processing")):
        # setup box
        logger.debug(f"Processing {i}/{total} grid tiles.")
        bbox_west = int(lon)
        bbox_east = int(lon + bbox_size_lon)
        bbox_south = int(lat)
        bbox_north = int(lat + bbox_size_lat)

        print(f"Run processing for bbox: {bbox_west} {bbox_east} {bbox_south} {bbox_north}")

        # setup outfile_download and check if already processed
        outfile_download = current_download_dir / f'DW_{ANALYSIS_DATE}_{bbox_west}_{bbox_east}_{bbox_south}_{bbox_north}.nc'
        outfile_breaks = current_breakpoint_dir / f'DW_{ANALYSIS_DATE}_{bbox_west}_{bbox_east}_{bbox_south}_{bbox_north}_breaks.parquet'

        # check if breakpoint file already exists
        if outfile_breaks.exists():
            print(f'Breakpoints have been already calculated!: Skip processing for {bbox_west} {bbox_south} \n')
            print('Data is loaded and appended \n')
            breaks_list.append(pd.read_parquet(outfile_breaks))

            # Periodically save partial results
            if len(breaks_list) >= 10:
                if breaks_list:
                    temp_merged = pd.concat(breaks_list, ignore_index=True)
                    temp_joined = gdf.set_index('id_geohash').join(temp_merged, how='inner').reset_index()
                    partial_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}_partial.parquet'
                    temp_joined.to_parquet(partial_file)
                    logger.info(f"Saved partial results to {partial_file}")
                    partial_saved = True
                    del temp_merged, temp_joined
                    gc.collect()
            continue

        # subset lakes to grid cell
        gdf_subset = filter_gdf_by_bbox(gdf=gdf, bbox_west=lon, bbox_east=lon + bbox_size_lon, bbox_south=lat,
                                        bbox_north=lat + bbox_size_lat)
        n_lakes = len(gdf_subset)
        print('Number of lakes: ', n_lakes)

        # extract lake ids
        id_list = gdf_subset['id_geohash'].values.tolist()
        if n_lakes == 0:
            print(f'No lakes available for grid {bbox_west} {bbox_south}. Skipping this grid cell! \n')
            continue

        # download
        if not outfile_download.exists():
            try:
                ds_dl = downloader.download_dw_monthly(gdf=gdf_subset, max_total_requests=2000, n_parallel=2,
                                                       date_list=[ANALYSIS_DATE], save_to_file=outfile_download)
            except ValueError as e:
                expected_msg = "No data was extracted from any chunk. Check GEE request parameters."
                if str(e) == expected_msg:
                    print(f"Expected error caught: {e}")
                    continue
                else:
                    raise
        else:
            print(f'Outfile already exists: Skipping download for {bbox_west} {bbox_south} \n')
            # If download already exists, load it for processing
            ds_dl = xr.open_dataset(outfile_download)

        # ========== CRITICAL FIX: Load historical data per tile, not globally ==========
        logger.info(f"Loading historical dataset for tile {i}...")
        ds_historical = xr.open_dataset(path_historical_dw)
        log_memory_usage(f"After loading historical dataset for tile {i}")

        # subset historical data to grid cell
        ds_historical_subset = ds_historical.sel(id_geohash=id_list)

        # Close historical immediately after subsetting to free memory
        close_and_clean(ds_historical, f"ds_historical_tile_{i}")

        # merge historical and recent and convert to DWDataset object
        ds_merged = xr.merge([ds_historical_subset, ds_dl]).sortby('date')
        dwds = DWDataset(ds_merged)

        # run breakpoint analysis
        breaks = bp.calculate_break(dataset=dwds, analysis_date=ANALYSIS_DATE)
        breaks.to_parquet(outfile_breaks)

        # add to merge list
        breaks_list.append(breaks)

        # Periodically save and clear breaks_list to prevent memory buildup
        if len(breaks_list) >= 10:
            logger.info(f"Saving intermediate results after {len(breaks_list)} tiles...")
            if breaks_list:
                breaks_merged = pd.concat(breaks_list, ignore_index=True)
                joined = gdf.set_index('id_geohash').join(breaks_merged, how='inner').reset_index()
                path_to_joined_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}_partial.parquet'
                joined.to_parquet(path_to_joined_file)
                logger.info(f"Saved partial results to {path_to_joined_file}")
                partial_saved = True
                breaks_list = []  # Reset list to free memory
                gc.collect()

        # Clean up to prevent memory buildup
        close_and_clean(ds_dl, f"ds_dl_{bbox_west}")
        close_and_clean(ds_historical_subset, f"ds_historical_subset_{bbox_west}")
        close_and_clean(ds_merged, f"ds_merged_{bbox_west}")

        # Force garbage collection after each tile
        gc.collect()
        log_memory_usage(f"After tile {i} cleanup")

    # Final concatenation of remaining breaks
    if breaks_list:
        breaks_merged = pd.concat(breaks_list, ignore_index=True)
        joined = gdf.set_index('id_geohash').join(breaks_merged, how='inner').reset_index()
        path_to_joined_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}.parquet'
        joined.to_parquet(path_to_joined_file)
        logger.info(f"Final combined file saved to {path_to_joined_file}")
    elif partial_saved:
        # Load the partial file if it exists and no breaks_list
        partial_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}_partial.parquet'
        if partial_file.exists():
            logger.info(f"Loading partial results from {partial_file}")
            final_joined = pd.read_parquet(partial_file)
            path_to_joined_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}.parquet'
            final_joined.to_parquet(path_to_joined_file)
            logger.info(f"Final combined file saved to {path_to_joined_file}")

    end = datetime.datetime.now()
    logger.debug(f"Finished processing {ANALYSIS_DATE} at time {end}")
    total_time = end - start
    logger.debug(f"Finished in {total_time}")

    logger.info(f"Combining historical and new DW data into a single netcdf file for {ANALYSIS_DATE}")

    # Get all downloaded files
    downloaded_files = sorted(glob.glob(str(current_download_dir / f'DW_{ANALYSIS_DATE}_*.nc')))
    output_netcdf = Path(output_dir) / f'lakes_dw_Vd2_{ANALYSIS_DATE}.nc'

    if downloaded_files:
        logger.info(f"Found {len(downloaded_files)} NetCDF files to combine")
        logger.info("Using memory-efficient batch processing...")

        # Load historical file for final merge
        logger.info("Loading historical dataset for final merge...")
        ds_historical = xr.open_dataset(most_recent_dynamic_world_file)
        log_memory_usage("After loading historical dataset for final merge")

        # Process in batches to avoid memory issues
        BATCH_SIZE = 2  # Process 2 files at a time for better memory management
        combined = None
        total_duplicates_removed = 0

        num_batches = (len(downloaded_files) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in tqdm(range(0, len(downloaded_files), BATCH_SIZE), desc="Processing batches",
                              total=num_batches):
            batch_files = downloaded_files[batch_idx:batch_idx + BATCH_SIZE]
            logger.info(f"Processing batch {batch_idx // BATCH_SIZE + 1}/{num_batches} ({len(batch_files)} files)")
            log_memory_usage(f"Before batch {batch_idx // BATCH_SIZE + 1}")

            # Load this batch
            batch_datasets = []
            for nc_file in batch_files:
                ds = xr.open_dataset(nc_file)
                batch_datasets.append(ds)

            # Concatenate this batch
            batch_combined = xr.concat(batch_datasets, dim='id_geohash')

            # Remove duplicates within batch
            _, unique_idx = np.unique(batch_combined['id_geohash'].values, return_index=True)
            unique_idx = np.sort(unique_idx)
            if len(unique_idx) < len(batch_combined['id_geohash']):
                dup_count = len(batch_combined['id_geohash']) - len(unique_idx)
                total_duplicates_removed += dup_count
                logger.debug(f"  Removed {dup_count} duplicates in this batch")
                batch_combined = batch_combined.isel(id_geohash=unique_idx)

            # Sort this batch
            batch_combined = batch_combined.sortby('id_geohash')

            # Merge with previous combined data
            if combined is None:
                combined = batch_combined
            else:
                # Concatenate with previous combined
                combined = xr.concat([combined, batch_combined], dim='id_geohash')
                # Remove duplicates
                _, unique_idx = np.unique(combined['id_geohash'].values, return_index=True)
                unique_idx = np.sort(unique_idx)
                if len(unique_idx) < len(combined['id_geohash']):
                    dup_count = len(combined['id_geohash']) - len(unique_idx)
                    total_duplicates_removed += dup_count
                    logger.debug(f"  Removed {dup_count} duplicates during merge")
                    combined = combined.isel(id_geohash=unique_idx)
                # Sort after merge
                combined = combined.sortby('id_geohash')

            # Close batch datasets to free memory
            for ds in batch_datasets:
                ds.close()
            batch_datasets.clear()

            # Force garbage collection
            gc.collect()
            log_memory_usage(f"After batch {batch_idx // BATCH_SIZE + 1}")

        if total_duplicates_removed > 0:
            logger.info(f"Total duplicates removed: {total_duplicates_removed}")

        if combined is not None:
            # Final sort to ensure everything is ordered
            logger.info("Final sorting of combined data...")
            combined = combined.sortby('id_geohash')
            log_memory_usage("Before merging with historical")

            # Use chunked merging
            logger.info("Merging with historical data using chunked approach...")
            merge_netcdf_chunked(ds_historical, combined, output_netcdf, chunk_size=250)

            # Clean up the combined dataset
            close_and_clean(combined, "combined")

            # Verify the output file
            if output_netcdf.exists():
                file_size_gb = output_netcdf.stat().st_size / (1024 ** 3)
                logger.info(f"Successfully created combined netcdf: {output_netcdf}")
                logger.info(f"File size: {file_size_gb:.2f} GB")
            else:
                logger.error("Failed to create combined netcdf")

            # Close historical dataset
            close_and_clean(ds_historical, "ds_historical")

        else:
            logger.warning("No data was combined")
    else:
        logger.warning(f"No downloaded files found in {current_download_dir}")

    logger.info("Combining breakpoint parquet files")

    # Get all breakpoint parquet files
    break_files = sorted(glob.glob(str(current_breakpoint_dir / f'DW_{ANALYSIS_DATE}_*_breaks.parquet')))

    if break_files:
        dfs = []
        total_rows = 0

        for file in tqdm(break_files, desc="Reading breakpoint files"):
            df = pd.read_parquet(file)
            dfs.append(df)
            total_rows += len(df)
            logger.debug(f"Read {file} with {len(df)} rows. Total rows so far: {total_rows}")

            # Clear memory periodically
            if len(dfs) >= 20:
                temp_combined = pd.concat(dfs, ignore_index=True)
                dfs = [temp_combined]
                gc.collect()

        if dfs:
            breaks_combined = pd.concat(dfs, ignore_index=True)
            breaks_combined = breaks_combined.sort_values('drainage_confidence', ascending=False)

            output_parquet = Path(
                output_dir) / f'DW_{ANALYSIS_DATE}_{X_MIN_START}_{X_MIN_END}_{Y_MIN_START}_{Y_MIN_END}.parquet'

            breaks_combined.to_parquet(output_parquet, index=False)

            logger.info(f"Successfully created combined parquet: {output_parquet}")
            logger.info(f"Total rows: {len(breaks_combined)}")
            logger.info(f"File size: {output_parquet.stat().st_size / (1024 ** 2):.2f} MB")

            del dfs, breaks_combined
        else:
            logger.warning("No data found in breakpoint files")
    else:
        logger.warning(f"No breakpoint files found in {current_breakpoint_dir}")

    log_memory_usage("Program end")
    logger.info("Script completed successfully")


if __name__ == '__main__':
    main()