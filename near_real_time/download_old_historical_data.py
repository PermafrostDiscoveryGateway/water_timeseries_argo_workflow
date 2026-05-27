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

    # TODO get new lake ids here
    # ===== GET NEW LAKE IDS =====
    print("\n" + "=" * 70)
    print("STEP 1: Finding new lake_ids (in new data but not in old data)")
    print("=" * 70)

    # Collect all new lake_ids from streaming generator
    all_new_lake_ids = []
    for batch in get_new_lake_ids_streaming(
            earliest_dynamic_world_file,
            dynamic_world_dir,
            chunk_size=CHUNK_SIZE
    ):
        all_new_lake_ids.extend(batch)
        print(f"  Accumulated {len(all_new_lake_ids):,} new lake_ids so far...")

    print(f"\n✓ Total new lake_ids found: {len(all_new_lake_ids):,}")

    # Optional: Save to file for reference
    new_lakes_file = os.path.join(missing_historical_data_dir, 'new_lake_ids.txt')
    with open(new_lakes_file, 'w') as f:
        for lake_id in all_new_lake_ids:
            f.write(f"{lake_id}\n")
    print(f"✓ Saved new lake_ids to {new_lakes_file}")

    # Verify all new lake_ids exist in vector file
    vector_ids = set(full_gdf['id_geohash'].values)
    new_ids_set = set(all_new_lake_ids)
    missing_from_vector = new_ids_set - vector_ids

    if missing_from_vector:
        print(f"\n⚠️ WARNING: {len(missing_from_vector)} lake_ids missing from vector file!")
        print(f"   Sample missing: {list(missing_from_vector)[:10]}")
        print(f"   These lakes cannot be downloaded - filtering them out.")

        # Filter to only IDs that exist in vector file
        all_new_lake_ids = [lid for lid in all_new_lake_ids if lid in vector_ids]
        print(f"   Remaining valid lake_ids: {len(all_new_lake_ids):,}")
    else:
        print(f"\n✓ All {len(all_new_lake_ids):,} new lake_ids found in vector file!")

    # Filter to new lakes
    print("\n" + "=" * 70)
    print("STEP 2: Filtering GeoDataFrame to target lakes")
    print("=" * 70)
    print("Filtering to target lakes...")
    target_gdf = full_gdf[full_gdf['id_geohash'].isin(all_new_lake_ids)].copy()
    print(f"✓ Filtered GeoDataFrame has {len(target_gdf):,} lakes")

    # Optional: Clear full_gdf from memory to save space
    del full_gdf
    import gc

    gc.collect()
    print("✓ Cleared full vector file from memory")

    print(f"\nReady to download historical data for {len(target_gdf)} lakes")
    print(f"Total downloads needed: {len(earlier_years) * len(months)} files")

    # TODO do not use below, wrong naming scheme

    # ===== DOWNLOAD HISTORICAL DATA =====
    print("\n" + "=" * 70)
    print("STEP 3: Downloading historical data")
    print("=" * 70)

    # Create a timestamped directory for this run
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = os.path.join(missing_historical_data_dir, f"download_run_{current_time}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"✓ Created run directory: {run_dir}")

    # Create a single downloader instance (reuse it)
    dl = EarthEngineDownloader(ee_auth=True, logger=logger)

    # Track progress
    total_downloads = len(earlier_years) * len(months)
    completed_downloads = 0

    for year in earlier_years:
        for month in months:
            completed_downloads += 1
            filename = f"historical_{year}_{month:02d}.nc"
            filepath = os.path.join(run_dir, filename)

            print(f"\n[{completed_downloads}/{total_downloads}] Processing: {year}-{month:02d}")

            # Check if file already exists
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                print(f"  ✓ File already exists, skipping: {filename}")
                continue

            try:
                print(f"  Downloading {filename}...")
                ds = dl.download_dw_monthly(
                    gdf=target_gdf,  # Use the filtered GeoDataFrame
                    name_attribute="id_geohash",
                    id_list=all_new_lake_ids,
                    years=[year],
                    months=[month],
                    save_to_file=filepath,
                    max_total_requests=500,
                    n_parallel=1,
                )

                print(f"  ✓ Successfully downloaded {filename}")

                # Clear memory after each download
                if ds is not None:
                    del ds
                gc.collect()

            except Exception as e:
                logger.error(f"  ✗ Error downloading {year}-{month}: {e}")
                # Remove partial/corrupt file
                if os.path.exists(filepath) and os.path.getsize(filepath) == 0:
                    os.remove(filepath)
                continue

    print("\n" + "=" * 70)
    print("✓ ALL DOWNLOADS COMPLETE!")
    print(f"✓ Data saved to: {run_dir}")
    print("=" * 70)