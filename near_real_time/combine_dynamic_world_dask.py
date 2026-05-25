import pandas as pd
from datetime import datetime
from loguru import logger
import os
import numpy as np
import glob
import xarray as xr
from dotenv import load_dotenv
import dask
from dask.distributed import LocalCluster, Client
import gc


def combine_new_dynamic_world_data_with_latest(env_path=None):
    # Load environment
    if env_path is None:
        load_dotenv()
        logger.info("Loading environment from default .env file")
    else:
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")

    # Get environment variables
    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID
    dynamic_world_dir = os.environ['dynamic_world_dir']
    split_new_dynamic_world_data_dir = os.environ['split_new_dynamic_world_data_dir']

    # Initialize Dask client with memory limits
    logger.info("Initializing Dask cluster...")
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=2,
        memory_limit='12GB',
        dashboard_address=None
    )
    client = Client(cluster)
    logger.info(f"Dask client initialized: {client}")

    try:
        # Find latest valid file
        all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
        valid_files = []
        for f in all_dynamic_world_files:
            if os.path.getsize(f) > 1024 * 1024:
                valid_files.append(f)
            else:
                logger.warning(f"Skipping empty/corrupted file: {os.path.basename(f)}")

        if not valid_files:
            logger.error("No valid existing dynamic world files found!")
            return None

        most_recent_dynamic_world_file = max(valid_files, key=os.path.getctime)
        logger.info(f"Most recent dynamic world file: {most_recent_dynamic_world_file}")
        logger.info(f"File size: {os.path.getsize(most_recent_dynamic_world_file) / (1024 ** 3):.2f} GB")

        # Get downloaded chunk files
        downloaded_files = glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc"))
        logger.info(f"Found {len(downloaded_files)} chunk files")

        # Validate chunk files
        valid_chunks = []
        for chunk_file in downloaded_files:
            try:
                with xr.open_dataset(chunk_file, decode_times=False) as test:
                    test.close()
                valid_chunks.append(chunk_file)
            except Exception as e:
                logger.warning(f"Skipping corrupted chunk: {os.path.basename(chunk_file)} - {str(e)[:50]}")

        logger.info(f"Valid chunks: {len(valid_chunks)} out of {len(downloaded_files)}")

        if not valid_chunks:
            logger.error("No valid chunk files found!")
            return None

        # Read and combine new data chunks using Dask
        logger.info("Reading and combining new data chunks with Dask (lazy loading)...")

        # Process chunks in batches
        batch_size = 20
        merged_new_ds = None

        for batch_start in range(0, len(valid_chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_chunks))
            logger.info(f"Processing chunks {batch_start + 1}-{batch_end}/{len(valid_chunks)}...")

            batch_datasets = []
            for chunk_file in valid_chunks[batch_start:batch_end]:
                try:
                    ds_chunk = xr.open_dataset(
                        chunk_file,
                        decode_times=False,
                        chunks={'id_geohash': 10000, 'date': -1}
                    )

                    # Convert dates
                    if 'date' in ds_chunk.variables:
                        days_offset = 3653
                        ds_chunk['date'] = ds_chunk['date'] + days_offset
                        ds_chunk['date'].attrs['units'] = 'days since 2015-07-01'
                        ds_chunk['date'].attrs['calendar'] = 'proleptic_gregorian'

                    batch_datasets.append(ds_chunk)

                except Exception as e:
                    logger.error(f"Failed to load chunk {os.path.basename(chunk_file)}: {e}")
                    continue

            if batch_datasets:
                if len(batch_datasets) == 1:
                    batch_combined = batch_datasets[0]
                else:
                    batch_combined = xr.concat(batch_datasets, dim="id_geohash")

                if merged_new_ds is None:
                    merged_new_ds = batch_combined
                else:
                    merged_new_ds = xr.concat([merged_new_ds, batch_combined], dim="id_geohash")

            gc.collect()

        if merged_new_ds is None:
            logger.error("No data to merge from chunks")
            return None

        # Remove duplicate lake IDs
        logger.info("Removing duplicate lake IDs...")
        # Get the Dask array and compute it
        id_geohash_da = merged_new_ds.id_geohash.data  # This returns a Dask array
        id_geohash_values = id_geohash_da.compute()  # Now compute it

        _, unique_indices = np.unique(id_geohash_values, return_index=True)
        merged_new_ds = merged_new_ds.isel(id_geohash=sorted(unique_indices))

        # Re-chunk for optimal processing
        merged_new_ds = merged_new_ds.chunk({'id_geohash': 50000, 'date': -1})

        # Get dates
        logger.info("Getting date information...")
        dates = merged_new_ds.date.data.compute()
        latest_date = max(dates)

        # Convert to actual date for filename
        ref_date = datetime(2015, 7, 1)
        latest_date_obj = ref_date + pd.Timedelta(days=int(latest_date))
        latest_date_string = latest_date_obj.strftime('%Y_%m_%d')

        logger.info(f"Latest date (days since 2015-07-01): {latest_date}")
        logger.info(f"Latest date (actual): {latest_date_string}")
        logger.info(f"New data: {len(merged_new_ds.id_geohash):,} lakes x {len(dates)} dates")

        # Load existing dataset with Dask
        logger.info("Loading existing dataset with Dask...")
        existing_ds = xr.open_dataset(
            most_recent_dynamic_world_file,
            decode_times=False,
            chunks={'id_geohash': 50000, 'date': -1}
        )

        # Get existing dates
        existing_dates = existing_ds.date.data.compute()
        logger.info(f"Existing dates: {len(existing_dates)} dates")

        existing_date_objs = [ref_date + pd.Timedelta(days=int(d)) for d in existing_dates]
        logger.info(f"Date range: {min(existing_date_objs).date()} to {max(existing_date_objs).date()}")

        # Remove overlapping dates
        new_dates_set = set(dates)
        existing_dates_set = set(existing_dates)
        overlapping_dates = existing_dates_set & new_dates_set

        if overlapping_dates:
            logger.warning(f"Removing {len(overlapping_dates)} overlapping dates: {sorted(overlapping_dates)[:5]}...")
            overlapping_list = list(overlapping_dates)
            # Create a boolean mask for selection
            mask = xr.zeros_like(merged_new_ds.date, dtype=bool)
            for od in overlapping_list:
                mask = mask | (merged_new_ds.date == od)
            merged_new_ds = merged_new_ds.where(~mask, drop=True)
            dates = merged_new_ds.date.data.compute()

        if len(dates) == 0:
            logger.warning("No new dates to add")
            return None

        logger.info(f"Adding {len(dates)} new dates")
        new_date_objs = [ref_date + pd.Timedelta(days=int(d)) for d in dates]
        logger.info(f"New date range: {min(new_date_objs).date()} to {max(new_date_objs).date()}")

        # Merge datasets
        logger.info("Merging datasets...")
        final_ds = xr.concat([existing_ds, merged_new_ds], dim="date")
        final_ds = final_ds.sortby("date")

        # Re-chunk for efficient writing
        final_ds = final_ds.chunk({'id_geohash': 50000, 'date': -1})

        # Create output filename
        new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
        new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

        logger.info(f"Saving to: {new_dynamic_world_data_file}")
        logger.info("Writing to disk (this may take 30-60 minutes)...")

        # Setup compression encoding
        encoding = {var_name: {'zlib': True, 'complevel': 5}
                    for var_name in final_ds.data_vars if var_name not in ['date', 'id_geohash']}

        # Write to netCDF
        final_ds.to_netcdf(new_dynamic_world_data_file, encoding=encoding)

        # Get final statistics
        logger.info("Verifying output...")
        result_ds = xr.open_dataset(new_dynamic_world_data_file, decode_times=False)
        final_lake_count = len(result_ds.id_geohash)
        final_date_count = len(result_ds.date)
        result_ds.close()

        logger.info("=" * 60)
        logger.info(f"✓ MERGE COMPLETE!")
        logger.info(f"  Output: {new_dynamic_world_data_file}")
        logger.info(f"  Total lakes: {final_lake_count:,}")
        logger.info(f"  Total dates: {final_date_count}")
        logger.info(f"  New lakes: {final_lake_count - len(existing_ds.id_geohash):,}")
        logger.info(f"  New dates: {final_date_count - len(existing_dates):,}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        logger.info("Cleaning up Dask client...")
        client.close()
        cluster.close()
        gc.collect()

    return new_dynamic_world_data_file


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()