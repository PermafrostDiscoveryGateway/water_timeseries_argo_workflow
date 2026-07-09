from near_real_time_grid_v2 import verify_downloads_complete, verify_process_complete, merge_near_real_time_region, \
    process_near_real_time_region_dates_zarr, download_near_real_time_region_dates, generate_expected_dates, \
    merge_near_real_time_region_v3_simple, \
    compare_netcdf_files, verify_merged_netcdf, verify_merged_data, merge_near_real_time_region_v3_smart, \
    enable_memory_tracking, log_memory_usage, merge_near_real_time_region_v3_smart_local_disk
import sys
import shutil
import gc
import utils.download_new_dynamic_world_data as download_new_dynamic_world_data
from loguru import logger
from datetime import date, datetime
from dotenv import load_dotenv
import os
import glob
import time
import pandas as pd
import utils.region_boundaries
from pathlib import Path
import xarray as xr
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def get_creation_time(filepath):
    """Get file creation time on Linux (birth time) if available"""
    stat_info = os.stat(Path(filepath))
    try:
        # st_birthtime is the actual creation time on Linux
        creation_time = stat_info.st_birthtime
    except AttributeError:
        # Fallback to ctime if birthtime not available
        creation_time = stat_info.st_ctime
    return creation_time


# After merging, compare original vs new
def verify_merge_result(original_file, merged_file):
    """
    Compare original and merged files and log the results.
    """
    logger.info(f"\n{'=' * 80}")
    logger.info("VERIFYING MERGE RESULT")
    logger.info(f"{'=' * 80}")
    logger.info(f"Original: {original_file}")
    logger.info(f"Merged:   {merged_file}")

    result = compare_netcdf_files(
        file1_path=original_file,
        file2_path=merged_file,
        sample_ids=5,
        verbose=True
    )

    # Log summary
    if result['summary']['successful']:
        logger.info("✅ MERGE VERIFICATION PASSED")
        logger.info(f"   Size: {result['summary']['file1_size_gb']:.2f}GB → {result['summary']['file2_size_gb']:.2f}GB")
        if result['summary'].get('new_dates_added', 0) > 0:
            logger.info(f"   New dates added: {result['summary']['new_dates_added']}")
        if result['summary'].get('new_ids_added', 0) > 0:
            logger.info(f"   New IDs added: {result['summary']['new_ids_added']}")
    else:
        logger.error("❌ MERGE VERIFICATION FAILED")
        for issue in result['summary']['issues']:
            logger.error(f"   Issue: {issue}")

    return result


def is_file_ready(filepath, wait_seconds=0.5, checks=10):
    sizes = []
    for _ in range(checks):
        size = os.path.getsize(filepath)
        sizes.append(size)
        time.sleep(wait_seconds)

    # If size hasn't changed, assume writing is done
    return len(set(sizes)) == 1


def create_empty_netcdf_with_structure(filepath, source_file=None):
    """
    Create an empty NetCDF file with the same structure as the source file.
    If source_file is not provided, creates a minimal structure.

    Args:
        filepath: Path to create the empty NetCDF file
        source_file: Optional source file to copy structure from

    Returns:
        Path: Path to the created file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # If file already exists, delete it to start fresh
    if filepath.exists():
        filepath.unlink()
        logger.info(f"Removed existing file: {filepath}")

    try:
        if source_file and Path(source_file).exists():
            # Copy structure from source file
            logger.info(f"Creating empty NetCDF with structure from: {source_file}")
            ds_source = xr.open_dataset(source_file)

            # Create empty dataset with same dimensions but no data
            empty_ds = xr.Dataset()

            # Copy coordinates
            for coord in ds_source.coords:
                if coord in ['id_geohash', 'date']:
                    # Keep the coordinate but with empty values
                    empty_ds[coord] = ds_source[coord]
                else:
                    # Copy other coordinates as is
                    empty_ds[coord] = ds_source[coord]

            # Copy variables (with empty data)
            for var_name in ds_source.data_vars:
                # Get the variable from source
                var = ds_source[var_name]
                # Create empty version with same dimensions
                empty_ds[var_name] = xr.DataArray(
                    data=np.full(var.shape, np.nan, dtype=var.dtype),
                    dims=var.dims,
                    attrs=var.attrs
                )

            # Copy global attributes (filter out unsupported types)
            for attr_name, attr_value in ds_source.attrs.items():
                # Skip boolean attributes (NetCDF doesn't support them)
                if isinstance(attr_value, bool):
                    logger.debug(f"Skipping boolean attribute '{attr_name}'")
                    continue
                # Skip bytes attributes (NetCDF doesn't support them)
                if isinstance(attr_value, bytes):
                    logger.debug(f"Skipping bytes attribute '{attr_name}'")
                    continue
                # Only copy supported types
                try:
                    empty_ds.attrs[attr_name] = attr_value
                except Exception as e:
                    logger.warning(f"Could not copy attribute '{attr_name}': {e}")

            # Add metadata to indicate this is an empty file
            empty_ds.attrs['empty'] = "True"  # Use string instead of boolean
            empty_ds.attrs['created_at'] = datetime.now().isoformat()
            empty_ds.attrs['status'] = "empty_placeholder"
            empty_ds.attrs['source_file'] = str(source_file)

            ds_source.close()

        else:
            # Create minimal structure
            logger.info("Creating minimal empty NetCDF structure")
            empty_ds = xr.Dataset()

            # Create empty dimensions
            empty_ds['id_geohash'] = xr.DataArray([], dims=('id_geohash',))
            empty_ds['date'] = xr.DataArray([], dims=('date',))

            # Add basic variables
            for var_name in ['water', 'trees', 'grass', 'built', 'crops',
                             'shrub_and_scrub', 'flooded_vegetation', 'bare', 'snow_and_ice']:
                empty_ds[var_name] = xr.DataArray(
                    data=np.full((0, 0), np.nan, dtype=np.float32),
                    dims=('id_geohash', 'date')
                )

            # Add metadata (using string values only)
            empty_ds.attrs['empty'] = "True"
            empty_ds.attrs['created_at'] = datetime.now().isoformat()
            empty_ds.attrs['description'] = 'Empty placeholder for local disk merge'
            empty_ds.attrs['status'] = 'empty'

        # Write the empty file with proper encoding
        encoding = {}
        for var in empty_ds.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 1,
                'shuffle': True
            }

        # Also ensure any boolean attributes are converted to strings
        for attr_name, attr_value in list(empty_ds.attrs.items()):
            if isinstance(attr_value, bool):
                empty_ds.attrs[attr_name] = str(attr_value)
            elif isinstance(attr_value, bytes):
                try:
                    empty_ds.attrs[attr_name] = attr_value.decode('utf-8')
                except:
                    empty_ds.attrs[attr_name] = str(attr_value)

        empty_ds.to_netcdf(filepath, encoding=encoding)
        empty_ds.close()

        logger.info(f"✅ Created empty NetCDF file: {filepath}")
        logger.info(f"  Size: {filepath.stat().st_size / (1024 ** 2):.2f} MB")

        return filepath

    except Exception as e:
        logger.error(f"Error creating empty NetCDF: {e}")
        # Clean up partial file if it exists
        if filepath.exists():
            try:
                filepath.unlink()
            except:
                pass
        raise


def main():
    logger.debug(f"Beginning historical run")
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    enable_memory_tracking()
    log_memory_usage("Program start")

    try:
        import dask
        dask.config.set(scheduler='threads')
        dask.config.set({'array.chunk-size': '128MiB'})
    except:
        pass

    SHOULD_RUN = False

    summer_months = [6, 7, 8, 9]
    dynamic_world_data_dir = os.environ['dynamic_world_data']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
    most_recent_dynamic_world_file = None
    most_recent_dynamic_world_file = os.path.join(dynamic_world_data_dir, 'lakes_dw_V2d_2016-2025.nc')
    # for file in all_dynamic_world_files:
    #     time_created = get_creation_time(file)
    #     readable_time = datetime.fromtimestamp(time_created)
    #     logger.debug(f"Netcdf file {file} has creation date of {readable_time}")
    #     most_recent_dynamic_world_file = max(all_dynamic_world_files, key=lambda f: Path(f).stat().st_mtime)
    # logger.debug(f"Most recent dynamic world file {most_recent_dynamic_world_file}")

    TODAY = datetime.now()
    TODAY_MONTH = TODAY.month
    if TODAY_MONTH - 1 in summer_months:
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 3:
            SHOULD_RUN = True
            logger.debug(f"TODAY_DAY: {TODAY_DAY} should we run and check: {SHOULD_RUN}")

    if SHOULD_RUN:
        date_to_run = [datetime(TODAY.year, TODAY_MONTH - 1, 1).strftime("%Y-%m")]
        dates_to_run_string = date_to_run[0].replace('-', '_')

        # ===== TODO 1: Create empty netcdf file here to be used for writing =====
        # Use local disk for temporary file
        local_temp_dir = Path("/tmp/merge_temp")
        local_temp_dir.mkdir(parents=True, exist_ok=True)

        # Create the temp file on local disk
        local_merge_file = local_temp_dir / f"merged_historical_{dates_to_run_string}.nc"
        logger.info(f"Creating empty NetCDF file on local disk: {local_merge_file}")

        # Create empty NetCDF with structure from the most recent file
        try:
            create_empty_netcdf_with_structure(
                filepath=local_merge_file,
                source_file=most_recent_dynamic_world_file
            )
        except Exception as e:
            logger.error(f"Failed to create empty NetCDF: {e}")
            sys.exit(1)

        # Final Filestore path
        name_of_final_merge_file = f"{dynamic_world_data_dir}/lakes_dw_Vdc_v2_{dates_to_run_string}.nc"
        logger.debug(f"New netcdf file will be {name_of_final_merge_file}")
        logger.debug(f"Checking if we should merge")
        logger.debug(f"Merge if {date_to_run} are downloaded for all regions")

        REGIONS = utils.region_boundaries.get_region_boundaries()
        REGION_NAMES = list(REGIONS.keys())

        regions_downloaded = 0
        regions_downloaded_names = []

        for region in REGION_NAMES:
            downloads_complete = verify_downloads_complete(region=region, analysis_dates=date_to_run)
            logger.debug(downloads_complete)
            summary = downloads_complete['summary']
            logger.debug(f"Total expected downloads {summary['total_expected_downloads']}")
            logger.debug(f"Total successful downloads {summary['total_successful_downloads']}")
            total_skipped_and_successful_downloads = summary['total_skipped_downloads'] + summary[
                'total_successful_downloads']
            total_expected_downloads = summary['total_expected_downloads']
            percent_downloaded = float(total_skipped_and_successful_downloads) / float(total_expected_downloads)
            logger.debug(f"Percent downloaded for {region}: {percent_downloaded}")
            logger.debug(f"Percent downloaded: {percent_downloaded}")
            if downloads_complete['complete'] or percent_downloaded > 0.99:
                regions_downloaded += 1
                regions_downloaded_names.append(region)

        logger.debug(f"How many regions are finished downloading?")
        logger.debug(f"{regions_downloaded} regions downloaded")
        logger.debug(f"These regions are fully downloaded")

        for region in regions_downloaded_names:
            logger.debug(region)

        successfully_merged_region_count = 0
        merged_files = []  # Track which files were merged

        if regions_downloaded == len(REGION_NAMES):
            for region in REGION_NAMES:
                logger.debug(f"Checking if we already merged region {region} for {date_to_run}")

            # Get the source file (the most recent historical file)
            all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
            source_file = max(all_dynamic_world_files, key=lambda f: Path(f).stat().st_mtime)
            logger.info(f"Using source file: {source_file}")

            for region in REGION_NAMES:
                logger.info(f"Merging region: {region}")
                logger.debug(f"Checking what is the most recent netcdf file")
                all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
                most_recent_dynamic_world_file = max(all_dynamic_world_files, key=lambda f: Path(f).stat().st_mtime)

                # ===== TODO 2: Use _local_disk method and the empty netcdf file from TODO 1 =====
                logger.info(f"Merging {region} into {local_merge_file}")
                log_memory_usage(f"Before merging {region}")

                merge_result = merge_near_real_time_region_v3_smart_local_disk(
                    region=region,
                    dates_to_merge=date_to_run,
                    input_file_path=local_merge_file,  # Append to the local file
                    env_path=env_path,
                    skip_if_already_merged=True,
                    verify_downloads_first=True,
                    temp_dir="/tmp/merge_temp",  # Use local disk for temp files
                    final_copy_path=None  # We'll copy at the end
                )

                if merge_result.get('success', False):
                    successfully_merged_region_count += 1
                    merged_files.append(region)
                    logger.info(f"✅ Merging was successful for {region}")

                    # Log the merge result
                    if 'file_path' in merge_result:
                        logger.info(f"  Local file: {merge_result['file_path']}")
                        logger.info(f"  IDs: {merge_result.get('id_count', 0):,}")
                        logger.info(f"  Dates: {merge_result.get('date_count', 0)}")
                        file_size = merge_result.get('file_size_gb', 0)
                        logger.info(f"  Size: {file_size:.2f} GB")

                    # Force garbage collection
                    gc.collect()
                    log_memory_usage(f"After merging region {region}")
                else:
                    logger.error(f"❌ Merge failed for {region}: {merge_result.get('error', 'Unknown error')}")

            logger.debug(f"Verifying merge finished for all regions properly")
        else:
            logger.warning(f"Not all regions downloaded, do not merge")
            logger.warning(f"Downloaded: {regions_downloaded}/{len(REGION_NAMES)}")
            missing_regions = [r for r in REGION_NAMES if r not in regions_downloaded_names]
            logger.warning(f"Missing regions: {missing_regions}")

        logger.debug(f"Successfully merged {successfully_merged_region_count} regions out of {len(REGION_NAMES)}")

        # ===== TODO 3: Copy the temp file from 1, now full, to the name of final merge file =====
        if successfully_merged_region_count == len(REGION_NAMES):
            logger.info("=" * 80)
            logger.info("✅ ALL REGIONS MERGED SUCCESSFULLY")
            logger.info("=" * 80)

            # Verify the local file exists and has data
            if local_merge_file.exists():
                file_size_gb = local_merge_file.stat().st_size / (1024 ** 3)
                logger.info(f"Local merge file size: {file_size_gb:.2f} GB")

                # Verify the file is valid
                try:
                    logger.info("Verifying local merge file...")
                    verify_ds = xr.open_dataset(local_merge_file)
                    id_count = len(verify_ds['id_geohash'])
                    date_count = len(verify_ds['date'])
                    verify_ds.close()
                    logger.info(f"✅ Local file is valid: {id_count:,} IDs, {date_count} dates")
                except Exception as e:
                    logger.error(f"❌ Local file verification failed: {e}")
                    sys.exit(1)

                # Check if target exists and create backup
                if Path(name_of_final_merge_file).exists():
                    backup_file = f"{name_of_final_merge_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    logger.info(f"Target file exists, backing up to: {backup_file}")
                    shutil.move(name_of_final_merge_file, backup_file)

                # Copy local file to Filestore
                logger.info(f"Copying {local_merge_file} to {name_of_final_merge_file}")
                start_time = time.time()

                # Use shutil.copy2 to preserve metadata
                shutil.copy2(local_merge_file, name_of_final_merge_file)

                copy_time = time.time() - start_time
                logger.info(f"✅ Copy completed in {copy_time:.2f} seconds")

                # Verify the copied file
                if Path(name_of_final_merge_file).exists():
                    final_size_gb = Path(name_of_final_merge_file).stat().st_size / (1024 ** 3)
                    logger.info(f"✅ Final file: {name_of_final_merge_file}")
                    logger.info(f"  Size: {final_size_gb:.2f} GB")

                    # Optional: Verify the copied file
                    try:
                        logger.info("Verifying final file...")
                        verify_ds = xr.open_dataset(name_of_final_merge_file)
                        id_count = len(verify_ds['id_geohash'])
                        date_count = len(verify_ds['date'])
                        verify_ds.close()
                        logger.info(f"✅ Final file is valid: {id_count:,} IDs, {date_count} dates")
                    except Exception as e:
                        logger.error(f"⚠️ Final file verification failed: {e}")
                else:
                    logger.error(f"❌ Failed to copy file to {name_of_final_merge_file}")
            else:
                logger.error(f"❌ Local merge file not found: {local_merge_file}")
        else:
            logger.warning("Not all regions merged successfully - keeping local file for debugging")
            logger.warning(f"Local file kept at: {local_merge_file}")

        # Clean up old temp files (keep last 5)
        try:
            temp_files = sorted(local_temp_dir.glob("merged_historical_*.nc"), key=lambda f: f.stat().st_mtime)
            if len(temp_files) > 5:
                for old_file in temp_files[:-5]:
                    logger.info(f"Removing old temp file: {old_file}")
                    old_file.unlink()
        except Exception as e:
            logger.warning(f"Could not clean up old temp files: {e}")

        logger.debug(f"Merging now finished")
        log_memory_usage("Program end")

    else:
        logger.debug("SHOULD_RUN is False - skipping merge")
        logger.debug(f"Current month: {TODAY_MONTH}, Summer months: {summer_months}")
        logger.debug(f"Condition: TODAY_MONTH - 1 in summer_months: {TODAY_MONTH - 1 in summer_months}")
        if TODAY_MONTH - 1 in summer_months:
            logger.debug(f"TODAY_DAY: {TODAY_DAY}, need > 3: {TODAY_DAY > 3}")


if __name__ == "__main__":
    main()