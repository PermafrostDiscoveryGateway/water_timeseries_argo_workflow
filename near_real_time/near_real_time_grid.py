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
from water_timeseries.downloader import EarthEngineDownloader
from water_timeseries.utils.spatial import create_longitude_latitude_grid, filter_gdf_by_bbox
from water_timeseries.dataset import DWDataset
from water_timeseries.breakpoint import NRTBreakpoint
import datetime
from utils.region_boundaries import get_region_boundaries
from utils.download_new_dynamic_world_data import download_new_dynamic_world_data, check_missing_data_in_netcdf
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


def main():
    # Set thread limits
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # os.environ['NUMEXPR_NUM_THREADS'] = '1'

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

    REGION_NAME = os.getenv("region_name", "TEST")

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

    hist_file_size_gb = get_file_size_gb(most_recent_dynamic_world_file)
    logger.info(f"Historical NetCDF file size: {hist_file_size_gb:.2f} GB")

    raw_missing_dates = check_missing_data_in_netcdf(most_recent_dynamic_world_file)
    missing_dates = []
    # ========== NEW LOGIC: Handle missing dates ==========
    if raw_missing_dates:
        logger.warning(f"Found {len(raw_missing_dates)} missing dates in historical data")
        for date in raw_missing_dates:
            missing_date_string = date.strftime("%Y-%m")
            logger.warning(f"Missing date: {missing_date_string}")
            missing_dates.append(missing_date_string)
        logger.info("Will download missing data and run breakpoint analysis")
        DOWNLOAD_REQUIRED = True
    else:
        logger.info("No missing dates found in historical data")
        logger.info("Will run breakpoint analysis using existing data only (no download)")
        DOWNLOAD_REQUIRED = False
    # TODO move this there

    vector_lake_file = os.environ['vector_lake_file']
    path_historical_dw = most_recent_dynamic_world_file
    path_lake_vector = vector_lake_file

    # ANALYSIS_DATE = "2026-05"
    for date in raw_missing_dates:
        date_string = date.strftime("%Y-%m")
        logger.info(f"Missing date: {date_string}")
        ANALYSIS_DATE = '2026-06-01'

        gdf = gpd.read_parquet(path_lake_vector)
        log_memory_usage("After loading lake vectors")

        bbox_size_lon = 1
        bbox_size_lat = 1
        grid = create_longitude_latitude_grid(lon_range=(X_MIN_START, X_MIN_END), lat_range=(Y_MIN_START, Y_MIN_END),
                                              bbox_size_lon=bbox_size_lon, bbox_size_lat=bbox_size_lat)
        print('created grid')
        log_memory_usage("After creating grid")

        bp = NRTBreakpoint()

        current_breakpoint_dir = Path(output_dir) / REGION_NAME / f'breakpoint_{ANALYSIS_DATE}'
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

        # run loop
        logger.debug(f"There are total {total} grid tiles for {REGION_NAME}")
        time.sleep(15)
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

            # ========== NEW LOGIC: Handle download vs no-download cases ==========
            if DOWNLOAD_REQUIRED:
                # Download or load existing file
                if not outfile_download.exists():
                    try:
                        ds_dl = downloader.download_dw_monthly(gdf=gdf_subset, max_total_requests=2000, n_parallel=2,
                                                               date_list=[ANALYSIS_DATE], save_to_file=str(outfile_download))
                    except ValueError as e:
                        if "No data was extracted" in str(e):
                            print(f"No data for {bbox_west} {bbox_south}")
                            continue
                        else:
                            raise
                else:
                    print(f'Loading existing download for {bbox_west} {bbox_south}')
                    ds_dl = xr.open_dataset(outfile_download)
            else:
                # No download required - we'll use historical data only
                # Create a dummy dataset with no new data, or simply use historical only
                print(f'No download needed for {bbox_west} {bbox_south} - using historical data only')
                # Create an empty dataset with the same structure but no new dates
                ds_dl = None

            # Load historical data for this tile
            logger.info(f"Loading historical dataset for tile {i}...")
            ds_historical = xr.open_dataset(path_historical_dw)

            # Subset historical data
            ds_historical_subset = ds_historical.sel(id_geohash=id_list)

            # Close historical immediately
            ds_historical.close()
            del ds_historical
            gc.collect()

            # ========== NEW LOGIC: Merge or use historical only ==========
            if DOWNLOAD_REQUIRED and ds_dl is not None:
                # Merge historical and new data
                ds_merged = xr.merge([ds_historical_subset, ds_dl]).sortby('date')
                # Clean up download dataset
                ds_dl.close()
                del ds_dl
            else:
                # Use only historical data
                logger.info(f"No new data to merge for grid {bbox_west} {bbox_south} - using historical data only")
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
        logger.debug(f"Finished processing in {end - start}")

        # ========== NEW LOGIC: Only combine NetCDF files if downloads occurred ==========
        if DOWNLOAD_REQUIRED:
            logger.info("Combining NetCDF files...")

            downloaded_files = sorted(glob.glob(str(current_download_dir / f'DW_{ANALYSIS_DATE}_*.nc')))
            output_netcdf = Path(output_dir) / f'lakes_dw_Vd2_{ANALYSIS_DATE}.nc'
            logger.debug(f"Output netcdf file being saved to {output_netcdf}")

            if downloaded_files:
                ds_historical = xr.open_dataset(most_recent_dynamic_world_file)

                BATCH_SIZE = 2
                combined = None

                for batch_idx in tqdm(range(0, len(downloaded_files), BATCH_SIZE), desc="Processing batches"):
                    batch_files = downloaded_files[batch_idx:batch_idx + BATCH_SIZE]
                    batch_datasets = []

                    for nc_file in batch_files:
                        ds = xr.open_dataset(nc_file)
                        batch_datasets.append(ds)

                    batch_combined = xr.concat(batch_datasets, dim='id_geohash')
                    _, unique_idx = np.unique(batch_combined['id_geohash'].values, return_index=True)
                    batch_combined = batch_combined.isel(id_geohash=np.sort(unique_idx))

                    if combined is None:
                        combined = batch_combined
                    else:
                        combined = xr.concat([combined, batch_combined], dim='id_geohash')
                        _, unique_idx = np.unique(combined['id_geohash'].values, return_index=True)
                        combined = combined.isel(id_geohash=np.sort(unique_idx))

                    for ds in batch_datasets:
                        ds.close()
                    gc.collect()

                if combined is not None:
                    merge_netcdf_chunked(ds_historical, combined, output_netcdf, chunk_size=250)
                    ds_historical.close()
        else:
            logger.info("No missing dates found - skipping NetCDF combination step (no new data to add)")

        logger.info("Script completed successfully")


if __name__ == '__main__':
    main()