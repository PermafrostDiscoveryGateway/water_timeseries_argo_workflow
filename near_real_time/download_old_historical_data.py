import netCDF4 as nc
import pandas as pd
from netCDF4 import num2date
from datetime import datetime
from loguru import logger
import os
import gc
import numpy as np
import glob
import xarray as xr
import geopandas as gpd
from dotenv import load_dotenv
from water_timeseries.downloader import EarthEngineDownloader
from water_timeseries.utils.io import load_vector_dataset


def get_new_lake_ids_streaming(old_file_path, new_dir_path, chunk_size=100000):
    """
    Memory-efficient version that streams through new files and yields
    new lake_ids in batches without storing all of them at once.
    """

    # Load old lake IDs (this is necessary but only ~hundreds of MB)
    print(f"Loading old file: {old_file_path}")
    with xr.open_dataset(old_file_path, decode_times=False) as old_ds:
        old_lake_ids = set(old_ds.id_geohash.values)
    print(f"Old file has {len(old_lake_ids):,} unique lake_ids")

    # Process new files
    chunk_files = sorted(glob.glob(os.path.join(new_dir_path, "*.nc")))
    print(f"Found {len(chunk_files)} chunk files")

    new_batch = []
    total_new = 0

    for i, chunk_file in enumerate(chunk_files):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(chunk_files)} chunks, "
                  f"new lakes found so far: {total_new:,}")

        try:
            with xr.open_dataset(chunk_file, decode_times=False) as ds:
                for lake_id in ds.id_geohash.values:
                    if lake_id not in old_lake_ids:
                        new_batch.append(lake_id)
                        total_new += 1

                        if len(new_batch) >= chunk_size:
                            yield new_batch
                            new_batch = []

        except Exception as e:
            print(f"Warning: Error reading {os.path.basename(chunk_file)}: {e}")

    if new_batch:
        yield new_batch

    print(f"Total new lake_ids found: {total_new:,}")


def check_existing_files(missing_historical_data_dir, chunk_num, years, months):
    """Check which year-month combinations already have files for a specific chunk."""
    files_to_download = []

    for year in years:
        for month in months:
            current_filename = f'historical_{year}_{month}_chunk_{chunk_num}.nc'
            current_filepath = os.path.join(missing_historical_data_dir, current_filename)

            if os.path.isfile(current_filepath):
                file_size = os.path.getsize(current_filepath)
                if file_size > 0:
                    print(f"  ✓ Skipping {current_filename} - already exists ({file_size / 1024 / 1024:.1f} MB)")
                else:
                    print(f"  ⚠ Warning: {current_filename} exists but is empty, will re-download")
                    files_to_download.append((year, month, current_filepath))
            else:
                print(f"  ✗ Need to download: {current_filename}")
                files_to_download.append((year, month, current_filepath))

    return files_to_download


if __name__ == "__main__":

    load_dotenv()

    vector_file = os.environ['vector_lake_file']
    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID

    dynamic_world_dir = os.environ['dynamic_world_dir']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
    earliest_dynamic_world_file = min(all_dynamic_world_files, key=os.path.getctime)

    missing_historical_data_dir = os.environ['missing_historical_data_dir']
    logger.debug(f"Missing historical data directory: {missing_historical_data_dir}")
    if not os.path.isdir(missing_historical_data_dir):
        os.makedirs(missing_historical_data_dir, exist_ok=True)

    current_time = str(datetime.now()).split(' ')[0].replace('-', '_')
    current_split_data_dir = os.path.join(missing_historical_data_dir, current_time)
    if not os.path.isdir(current_split_data_dir):
        os.makedirs(current_split_data_dir, exist_ok=True)

    logger.debug(f"Most recent dynamic world file: {most_recent_dynamic_world_file}")
    logger.debug(f"First dynamic world file: {earliest_dynamic_world_file}")

    # ===== CRITICAL: Load vector file ONCE with geometries =====
    print("Loading full vector file with geometries...")
    full_gdf = load_vector_dataset(vector_file, logger=logger)
    print(f"Loaded {len(full_gdf):,} total geometries")

    CHUNK_SIZE = 5000  # Adjust based on available RAM
    earlier_years = list(range(2016, 2025))
    months = [6, 7, 8, 9]

    print(f"Processing in chunks of {CHUNK_SIZE:,} lake_ids")
    print(f"Years to process: {earlier_years}")
    print(f"Months to process: {months}")

    # Process new lake_ids in chunks
    for chunk_num, new_lake_ids_chunk in enumerate(
            get_new_lake_ids_streaming(earliest_dynamic_world_file, dynamic_world_dir, chunk_size=CHUNK_SIZE),
            start=1
    ):
        print(f"\n{'=' * 70}")
        print(f"CHUNK {chunk_num} - Processing {len(new_lake_ids_chunk):,} lake_ids")
        print(f"{'=' * 70}")

        # Filter the GeoDataFrame to only include this chunk's lake_ids
        chunk_gdf = full_gdf[full_gdf['id_geohash'].isin(new_lake_ids_chunk)].copy()
        print(f"Found geometries for {len(chunk_gdf)}/{len(new_lake_ids_chunk)} lake_ids in this chunk")

        if len(chunk_gdf) == 0:
            print(f"Warning: No geometries found for this chunk's lake_ids. Skipping...")
            continue

        # Check existing files
        print("\nChecking existing files...")
        files_to_download = check_existing_files(
            current_split_data_dir,
            chunk_num,
            earlier_years,
            months
        )

        if not files_to_download:
            print(f"\n✓ All files for chunk {chunk_num} already exist. Skipping...")
            continue

        print(
            f"\nNeed to download {len(files_to_download)}/{len(earlier_years) * len(months)} files for chunk {chunk_num}")

        # Download missing files for this chunk
        for year, month, filepath in files_to_download:
            print(f"\nDownloading: historical_{year}_{month}_chunk_{chunk_num}.nc")

            try:
                dl = EarthEngineDownloader(ee_auth=True, logger=logger)

                # CRITICAL: Use gdf parameter instead of vector_dataset
                ds = dl.download_dw_monthly(
                    gdf=chunk_gdf,  # Pass the filtered GeoDataFrame directly
                    name_attribute="id_geohash",
                    id_list=list(new_lake_ids_chunk),  # Redundant but kept for safety
                    years=[year],
                    months=[month],
                    save_to_file=filepath,
                    max_total_requests=500,
                    n_parallel=1,
                )
                logger.info(f"✓ Successfully downloaded chunk {chunk_num} for {year}-{month}")

                # Clear Earth Engine memory if needed
                # Force cleanup
                if ds is not None:
                    del ds
                del dl
                gc.collect()

            except Exception as e:
                logger.error(f"✗ Error downloading chunk {chunk_num} for {year}-{month}: {e}")
                # Remove empty/corrupt file
                if os.path.exists(filepath) and os.path.getsize(filepath) == 0:
                    os.remove(filepath)

        # Create summary file
        summary_file = os.path.join(current_split_data_dir, f'chunk_{chunk_num}_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Chunk {chunk_num} completed at {datetime.now()}\n")
            f.write(f"Lake IDs in this chunk: {len(new_lake_ids_chunk):,}\n")
            f.write(f"Geometries found: {len(chunk_gdf)}\n")
            f.write(f"Years processed: {earlier_years}\n")
            f.write(f"Months processed: {months}\n")
            f.write(f"Files downloaded: {len(files_to_download)}\n")

        # Monitor memory
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"\nCurrent memory usage: {memory_mb:.1f} MB")
        except ImportError:
            pass

        print(f"\n✓ Chunk {chunk_num} complete!")

    print(f"\n{'=' * 70}")
    print("ALL CHUNKS COMPLETED")
    print(f"{'=' * 70}")