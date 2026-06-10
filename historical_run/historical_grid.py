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
from water_timeseries.breakpoint import NRTBreakpoint as OriginalNRTBreakpoint
import datetime
from utils.region_boundaries import get_region_boundaries
from utils.download_new_dynamic_world_data import download_new_dynamic_world_data, check_missing_data_in_netcdf
import resource
from joblib import Parallel, delayed


# ============= PATCHED NRTBREAKPOINT CLASS =============
class PatchedNRTBreakpoint(OriginalNRTBreakpoint):
    """Patched version of NRTBreakpoint that fixes the date column overlap issue."""

    def calculate_break(
            self,
            dataset,
            analysis_date,
            data_aggregation_period: str = "all",
            object_id=None,
            keep_nans: bool = False,
    ):
        """Calculate breakpoints with fixed join operation."""

        analysis_date = self._validate_analysis_date(analysis_date)
        print(analysis_date)
        print(analysis_date.strftime("%Y-%m"))

        # Check if analysis_date in dataset.dates_ (convert to YYYY-MM format for comparison)
        if analysis_date not in dataset.dates_:
            raise ValueError(f"Analysis date {analysis_date.strftime('%Y-%m')} is not available in the dataset.")

        # select dataset - default normalized data
        data = dataset.ds_normalized

        if object_id is not None:
            if isinstance(object_id, str):
                object_id = [object_id]
            object_id = [obj for obj in object_id if obj in dataset.object_ids_]
            data = data.sel(id_geohash=object_id)

        # split data into historical and analysis datasets based on analysis_date
        ds_analysis = data.sel(date=analysis_date)
        ds_historical = data.where(data["date"] < analysis_date, drop=True)

        if data_aggregation_period == "monthly":
            print("Filtering to monthly data for analysis date month:", analysis_date.month)
            ds_historical = ds_historical.where(ds_historical.date.dt.month == analysis_date.month, drop=True)

        # filter to dates where analysis date has some data
        ds_analysis_filtered, ds_historical_filtered, valid_ids, nan_ids = self._filter_valid_ids(
            ds_analysis, ds_historical
        )

        if len(valid_ids) == 0:
            if keep_nans:
                return pd.DataFrame(index=nan_ids, columns=self.output_columns)
            else:
                return pd.DataFrame(columns=self.output_columns)

        # loop over each lake and predict next value using ARIMA
        cpu_count = os.cpu_count() or 1
        n_jobs = max(1, min(cpu_count, len(valid_ids)))
        predictions = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self.predict_nrt_arima)(
                ds_in=ds_historical_filtered, id_geohash=idx, water_column=dataset.water_column
            )
            for idx in tqdm(valid_ids, desc="NRT breakpoints")
        )
        # remove None values
        predictions = [prediction for prediction in predictions if prediction is not None]
        prediction_df = pd.DataFrame(predictions)

        if prediction_df.empty:
            prediction_df = pd.DataFrame(
                index=ds_analysis_filtered.id_geohash.values,
                columns=self.output_columns,
            )

        # ========== FIXED JOIN OPERATION ==========
        # Convert the water column to dataframe and reset index
        left_df = ds_analysis_filtered[dataset.water_column].to_dataframe().reset_index()

        # Handle right_df - it has id_geohash as index name
        if prediction_df.index.name == 'id_geohash' or prediction_df.index.name is None:
            # Reset index to make id_geohash a column
            right_df = prediction_df.reset_index()
            # Rename the index column to 'id_geohash' if it's unnamed
            if 'index' in right_df.columns and 'id_geohash' not in right_df.columns:
                right_df = right_df.rename(columns={'index': 'id_geohash'})
        else:
            right_df = prediction_df.copy()

        # Ensure both dataframes have the required columns
        if 'id_geohash' not in left_df.columns:
            raise KeyError("left_df missing 'id_geohash' column")
        if 'id_geohash' not in right_df.columns:
            raise KeyError("right_df missing 'id_geohash' column")
        if 'date' not in left_df.columns:
            raise KeyError("left_df missing 'date' column")

        # Merge on id_geohash and date
        df_output = left_df.merge(right_df, on=['id_geohash', 'date'], how='left').round(4)
        # ==========================================

        # rename observed water column for clarity
        df_output.rename(columns={dataset.water_column: "water_observed"}, inplace=True)

        # calculate residuals
        df_output["water_residual"] = df_output["water_observed"] - df_output["water_predicted"]

        df_historical_stats = self._get_ds_stats(ds_historical_filtered, water_column=dataset.water_column).round(4)
        df_historical_stats.columns = "water_historical_" + df_historical_stats.columns.astype(str)

        df_output = df_output.join(df_historical_stats, on='id_geohash', how='left').round(4)

        # add confidence level to output
        df_output = self._add_confidence_level(df_output)

        # if keep_nans is selected: calculate historical stats for these and append to calculated data
        if keep_nans:
            prediction_df_nan = pd.DataFrame(
                index=nan_ids,
                columns=self.output_columns_base,
            )
            df_historical_stats_nans = self._get_ds_stats(
                ds_historical.sel(id_geohash=nan_ids), water_column=dataset.water_column
            ).round(4)
            df_historical_stats_nans.columns = "water_historical_" + df_historical_stats_nans.columns.astype(str)

            # Reset index for NaN data
            df_output_nan = prediction_df_nan.reset_index()
            df_output_nan = df_output_nan.rename(columns={'index': 'id_geohash'})
            df_output_nan = df_output_nan.join(df_historical_stats_nans, on='id_geohash', how='left').round(4)
            df_output = pd.concat([df_output, df_output_nan]).sort_values('id_geohash').reset_index(drop=True)

        # Set index back to id_geohash for consistency with expected output
        if 'id_geohash' in df_output.columns:
            df_output = df_output.set_index('id_geohash')

        # Select only the columns that exist in the output
        available_columns = [col for col in self.output_columns if col in df_output.columns]
        return df_output[available_columns]


# Replace the original NRTBreakpoint with our patched version
NRTBreakpoint = PatchedNRTBreakpoint


# ===================================================


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


def debug_dataframe_info(dwds, analysis_date):
    """Debug function to print dataframe info before breakpoint calculation"""
    print("\n" + "=" * 80)
    print(f"DEBUG INFO for analysis_date: {analysis_date}")
    print("=" * 80)

    # Check the water column
    water_col = dwds.water_column
    print(f"\n1. Water column name: '{water_col}'")

    # Get the water data as dataframe
    try:
        water_df = dwds.ds[water_col].to_dataframe()
        print(f"\n2. Water dataframe shape: {water_df.shape}")
        print(f"   Water dataframe index: {water_df.index.name}")
        print(f"   Water dataframe columns: {list(water_df.columns)}")
        print(f"   Water dataframe head:\n{water_df.head()}")
    except Exception as e:
        print(f"   Error getting water dataframe: {e}")

    # Check what calculate_break might be doing internally
    print(f"\n3. Dataset structure:")
    print(f"   Dimensions: {list(dwds.ds.dims.keys())}")
    print(f"   Coordinates: {list(dwds.ds.coords.keys())}")
    print(f"   Data variables: {list(dwds.ds.data_vars.keys())}")

    # Check if date is a coordinate or variable
    if 'date' in dwds.ds.coords:
        print(f"\n4. 'date' is a coordinate")
        print(f"   Date values: {dwds.ds['date'].values[:5]}...")
    elif 'date' in dwds.ds.data_vars:
        print(f"\n4. 'date' is a data variable")
        print(f"   Date shape: {dwds.ds['date'].shape}")
        print(f"   Date values: {dwds.ds['date'].values[:5]}...")
    else:
        print(f"\n4. 'date' not found in dataset")

    # Try to see what the breakpoint calculation will do
    print(f"\n5. Checking ds_analysis_filtered (from breakpoint.py):")
    # This simulates what happens inside calculate_break
    try:
        # Filter to analysis_date
        ds_filtered = dwds.ds.where(dwds.ds['date'] <= pd.to_datetime(analysis_date), drop=True)
        # Get the water column dataframe
        filtered_water_df = ds_filtered[water_col].to_dataframe()
        print(f"   Filtered water dataframe shape: {filtered_water_df.shape}")
        print(f"   Filtered water dataframe index: {filtered_water_df.index.names}")
        print(f"   Filtered water dataframe columns: {list(filtered_water_df.columns)}")
    except Exception as e:
        print(f"   Error simulating filter: {e}")

    print("=" * 80 + "\n")


def main():
    # Set thread limits
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

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

    # TODO get all the dates here
    dates_to_run = [datetime.date(year, month, 1).strftime("%Y-%m") for year in range(2016, 2026) for month in
                    [6, 7, 8, 9]]

    for date in dates_to_run:
        vector_lake_file = os.environ['vector_lake_file']
        path_historical_dw = most_recent_dynamic_world_file
        path_lake_vector = vector_lake_file

        ANALYSIS_DATE = date

        gdf = gpd.read_parquet(path_lake_vector)
        log_memory_usage("After loading lake vectors")

        bbox_size_lon = 1
        bbox_size_lat = 1
        grid = create_longitude_latitude_grid(lon_range=(X_MIN_START, X_MIN_END), lat_range=(Y_MIN_START, Y_MIN_END),
                                              bbox_size_lon=bbox_size_lon, bbox_size_lat=bbox_size_lat)
        print('created grid')
        log_memory_usage("After creating grid")

        # Use the patched NRTBreakpoint
        bp = NRTBreakpoint()

        current_breakpoint_dir = Path(output_dir) / f'breakpoint_{ANALYSIS_DATE}'
        current_breakpoint_dir.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Current breakpoint directory: {current_breakpoint_dir}")

        current_download_dir = Path(str(dynamic_world_download_dir), f'download_{ANALYSIS_DATE}')
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

        downloader = EarthEngineDownloader(ee_project=EE_PROJECT_ID)

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

            # ========== FIX: Filter IDs to only those that exist in historical data ==========
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

            # Download or load existing file
            if not outfile_download.exists():
                try:
                    ds_dl = downloader.download_dw_monthly(gdf=gdf_subset, max_total_requests=2000, n_parallel=2,
                                                           date_list=[ANALYSIS_DATE], save_to_file=outfile_download)
                except ValueError as e:
                    if "No data was extracted" in str(e):
                        print(f"No data for {bbox_west} {bbox_south}")
                        continue
                    else:
                        raise
            else:
                print(f'Loading existing download for {bbox_west} {bbox_south}')
                ds_dl = xr.open_dataset(outfile_download)

            # Load historical data for this tile only
            logger.info(f"Loading historical dataset for tile {i}...")
            ds_historical = xr.open_dataset(path_historical_dw)

            # Subset historical data (now with guaranteed valid IDs)
            ds_historical_subset = ds_historical.sel(id_geohash=id_list)

            # Close historical immediately
            ds_historical.close()
            del ds_historical
            gc.collect()

            # Merge and process
            ds_merged = xr.merge([ds_historical_subset, ds_dl], compat='override').sortby('date')
            dwds = DWDataset(ds_merged)

            breaks = bp.calculate_break(dataset=dwds, analysis_date=ANALYSIS_DATE)
            breaks.to_parquet(outfile_breaks)
            breaks_list.append(breaks)

            # Clean up
            ds_dl.close()
            ds_historical_subset.close()
            ds_merged.close()
            del ds_dl, ds_historical_subset, ds_merged
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

    logger.info("Script completed successfully")


if __name__ == '__main__':
    main()