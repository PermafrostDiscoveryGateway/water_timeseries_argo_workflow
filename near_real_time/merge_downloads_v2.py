from near_real_time_grid_v2 import verify_downloads_complete, verify_process_complete, merge_near_real_time_region, \
    process_near_real_time_region_dates_zarr, download_near_real_time_region_dates, generate_expected_dates, \
    merge_near_real_time_region_v3_simple,merge_near_real_time_region_v3_chunked,  \
    compare_netcdf_files, verify_merged_netcdf, verify_merged_data, merge_near_real_time_region_v3_smart, \
    enable_memory_tracking, log_memory_usage, merge_near_real_time_region_v3_smart_local_disk
import sys
import shutil
import gc
import utils.download_new_dynamic_world_data as download_new_dynamic_world_data
from loguru import logger
from datetime import date, datetime
from dotenv import load_dotenv
import subprocess
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

import subprocess


def copy_and_compress_netcdf(source_file: str, target_file: str, compression_level: int = 4):
    """
    Copy a NetCDF file with compression using nccopy.

    Args:
        source_file: Path to source NetCDF file
        target_file: Path to target (compressed) NetCDF file
        compression_level: Compression level (1-9, higher = more compression but slower)

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Copying and compressing {source_file} to {target_file}")
    logger.info(f"  Source size: {Path(source_file).stat().st_size / (1024 ** 3):.2f} GB")

    # Use nccopy with compression
    cmd = [
        'nccopy',
        '-d', str(compression_level),  # Compression level (1-9)
        '-s',  # Shuffle filter (improves compression)
        source_file,
        target_file
    ]

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.time() - start_time

        if Path(target_file).exists():
            target_size_gb = Path(target_file).stat().st_size / (1024 ** 3)
            compression_ratio = Path(source_file).stat().st_size / Path(target_file).stat().st_size
            logger.info(f"✅ Compression complete in {elapsed:.2f} seconds")
            logger.info(f"  Original: {Path(source_file).stat().st_size / (1024 ** 3):.2f} GB")
            logger.info(f"  Compressed: {target_size_gb:.2f} GB")
            logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
            return True
        else:
            logger.error("Target file not created")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"nccopy failed: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.warning("nccopy not found, falling back to xarray method")
        return copy_and_compress_netcdf_xarray(source_file, target_file, compression_level)


def copy_and_compress_netcdf_xarray(source_file: str, target_file: str, compression_level: int = 4):
    """
    Copy and compress NetCDF using xarray (fallback if nccopy not available).
    """
    logger.info(f"Using xarray to copy and compress: {source_file} -> {target_file}")

    try:
        start_time = time.time()

        # Open the source file (memory-mapped, doesn't load everything)
        ds = xr.open_dataset(source_file)

        # Create encoding with compression
        encoding = {}
        for var in ds.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': compression_level,
                'shuffle': True
            }

        # Write with compression
        ds.to_netcdf(target_file, encoding=encoding)
        ds.close()

        elapsed = time.time() - start_time

        if Path(target_file).exists():
            target_size_gb = Path(target_file).stat().st_size / (1024 ** 3)
            compression_ratio = Path(source_file).stat().st_size / Path(target_file).stat().st_size
            logger.info(f"✅ Compression complete in {elapsed:.2f} seconds")
            logger.info(f"  Original: {Path(source_file).stat().st_size / (1024 ** 3):.2f} GB")
            logger.info(f"  Compressed: {target_size_gb:.2f} GB")
            logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
            return True
        else:
            logger.error("Target file not created")
            return False

    except Exception as e:
        logger.error(f"xarray compression failed: {e}")
        return False


def copy_netcdf_with_progress(source_file: str, target_file: str, compression_level: int = 4):
    """
    Copy a NetCDF file with compression using the best available method.
    """
    # Try nccopy first (faster)
    try:
        import subprocess
        # Check if nccopy is available
        subprocess.run(['nccopy', '--version'], capture_output=True, check=True)
        return copy_and_compress_netcdf(source_file, target_file, compression_level)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("nccopy not available, using xarray method")
        return copy_and_compress_netcdf_xarray(source_file, target_file, compression_level)

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

    # Explicitly use the known good source file
    source_file = os.path.join(dynamic_world_data_dir, 'lakes_dw_V2d_2016-2025.nc')

    # Check if the source file exists
    if not Path(source_file).exists():
        logger.error(f"Source file not found: {source_file}")
        # Try to find any other .nc file as fallback
        if all_dynamic_world_files:
            valid_files = [f for f in all_dynamic_world_files
                           if 'temp' not in f and 'backup' not in f and 'merged_historical' not in f]
            if valid_files:
                source_file = max(valid_files, key=lambda f: Path(f).stat().st_mtime)
                logger.info(f"Using fallback source file: {source_file}")
            else:
                logger.error("No valid source files found")
                sys.exit(1)
        else:
            logger.error("No NetCDF files found")
            sys.exit(1)

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

        # ===== NEW: Check if we should use chunked or local disk merge =====
        # Check available local disk space
        import shutil
        disk_usage = shutil.disk_usage("/tmp")
        free_gb = disk_usage.free / (1024 ** 3)
        source_size_gb = Path(source_file).stat().st_size / (1024 ** 3)

        logger.info(f"Available local disk: {free_gb:.2f} GB")
        logger.info(f"Source file size: {source_size_gb:.2f} GB")

        # Use chunked merge if free space is limited or source file is large
        use_chunked_merge = (free_gb < 15 or source_size_gb > 8)
        logger.info(f"Using chunked merge: {use_chunked_merge}")

        # Create temp directories
        local_temp_dir = Path("/tmp/merge_temp")
        local_temp_dir.mkdir(parents=True, exist_ok=True)

        # Use local disk for temporary file
        local_merge_file = local_temp_dir / f"merged_historical_{dates_to_run_string}.nc"
        logger.info(f"Local merge file will be: {local_merge_file}")

        # Final Filestore path
        name_of_final_merge_file = f"{dynamic_world_data_dir}/lakes_dw_Vdc_v2_{dates_to_run_string}.nc"
        logger.debug(f"New netcdf file will be {name_of_final_merge_file}")

        # Verify downloads are complete
        REGIONS = utils.region_boundaries.get_region_boundaries()
        REGION_NAMES = list(REGIONS.keys())

        regions_downloaded = 0
        regions_downloaded_names = []

        for region in REGION_NAMES:
            downloads_complete = verify_downloads_complete(region=region, analysis_dates=date_to_run)
            logger.debug(downloads_complete)
            summary = downloads_complete['summary']
            logger.debug(f"Total expected downloads {summary['total_expected_downloads']}")
            total_skipped_and_successful_downloads = summary['total_skipped_downloads'] + summary[
                'total_successful_downloads']
            total_expected_downloads = summary['total_expected_downloads']
            percent_downloaded = float(total_skipped_and_successful_downloads) / float(
                total_expected_downloads) if total_expected_downloads > 0 else 0
            logger.debug(f"Percent downloaded for {region}: {percent_downloaded}")

            if downloads_complete['complete'] or percent_downloaded > 0.99:
                regions_downloaded += 1
                regions_downloaded_names.append(region)

        logger.debug(f"{regions_downloaded} regions downloaded")

        successfully_merged_region_count = 0

        if regions_downloaded == len(REGION_NAMES):
            for region in REGION_NAMES:
                logger.info(f"Processing region: {region}")
                log_memory_usage(f"Before processing {region}")

                if use_chunked_merge:
                    # ===== OPTION 1: Use chunked merge (memory efficient) =====
                    logger.info(f"Using chunked merge for {region}")

                    # Process each region with chunked merge
                    merge_result = merge_near_real_time_region_v3_chunked(
                        region=region,
                        dates_to_merge=date_to_run,
                        source_file=source_file,
                        output_file=local_merge_file,
                        env_path=env_path,
                        chunk_size=50000,  # Adjust based on memory
                        temp_dir="/tmp/merge_temp",
                    )
                else:
                    # ===== OPTION 3: Use local disk merge with per-region cleanup =====
                    logger.info(f"Using local disk merge for {region}")

                    # First, compress the source file if needed
                    compressed_source = local_temp_dir / f"compressed_source_{dates_to_run_string}.nc"
                    if not compressed_source.exists():
                        logger.info("Compressing source file...")
                        copy_and_compress_netcdf(source_file, compressed_source, compression_level=4)
                        logger.info(f"Compressed source: {compressed_source.stat().st_size / (1024 ** 3):.2f} GB")

                    # Merge region
                    merge_result = merge_near_real_time_region_v3_smart_local_disk(
                        region=region,
                        dates_to_merge=date_to_run,
                        input_file_path=local_merge_file,
                        env_path=env_path,
                        skip_if_already_merged=True,
                        temp_dir="/tmp/merge_temp",
                        final_copy_path=None,
                    )

                if merge_result.get('success', False):
                    successfully_merged_region_count += 1
                    logger.info(f"✅ Merging was successful for {region}")

                    # Log the merge result
                    if 'file_path' in merge_result:
                        logger.info(f"  Local file: {merge_result['file_path']}")
                        logger.info(f"  IDs: {merge_result.get('id_count', 0):,}")
                        logger.info(f"  Dates: {merge_result.get('date_count', 0)}")
                        file_size = merge_result.get('file_size_gb', 0)
                        logger.info(f"  Size: {file_size:.2f} GB")

                    # ===== CLEANUP after each region (Option 3) =====
                    # Force garbage collection
                    gc.collect()
                    log_memory_usage(f"After processing {region}")

                    # Check and clean up temp files
                    temp_files = list(local_temp_dir.glob("chunk_*.nc"))
                    if temp_files:
                        for f in temp_files:
                            try:
                                f.unlink()
                                logger.debug(f"Removed chunk file: {f}")
                            except:
                                pass

                    # Check disk usage
                    used_gb = shutil.disk_usage("/tmp").used / (1024 ** 3)
                    logger.info(f"Local disk usage after {region}: {used_gb:.2f} GB")

                    # If getting close to limit, recompress
                    if used_gb > 8 and local_merge_file.exists():
                        logger.info("Recompressing local file to free space...")
                        temp_recompress = local_temp_dir / f"recompress_{dates_to_run_string}.nc"
                        copy_and_compress_netcdf(local_merge_file, temp_recompress, compression_level=6)
                        shutil.move(temp_recompress, local_merge_file)
                        logger.info(f"Recompressed size: {local_merge_file.stat().st_size / (1024 ** 3):.2f} GB")
                else:
                    logger.error(f"❌ Merge failed for {region}: {merge_result.get('error', 'Unknown error')}")

            logger.debug(f"Verifying merge finished for all regions properly")
        else:
            logger.warning(f"Not all regions downloaded, do not merge")
            logger.warning(f"Downloaded: {regions_downloaded}/{len(REGION_NAMES)}")
            missing_regions = [r for r in REGION_NAMES if r not in regions_downloaded_names]
            logger.warning(f"Missing regions: {missing_regions}")

        logger.debug(f"Successfully merged {successfully_merged_region_count} regions out of {len(REGION_NAMES)}")

        # ===== Final copy to Filestore =====
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

                    # Verify the copied file
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
            temp_files = sorted(local_temp_dir.glob("*.nc"), key=lambda f: f.stat().st_mtime)
            if len(temp_files) > 5:
                for old_file in temp_files[:-5]:
                    if old_file != local_merge_file:
                        logger.info(f"Removing old temp file: {old_file}")
                        old_file.unlink()
        except Exception as e:
            logger.warning(f"Could not clean up old temp files: {e}")

        logger.debug(f"Merging now finished")
        log_memory_usage("Program end")

    else:
        logger.debug("SHOULD_RUN is False - skipping merge")


if __name__ == "__main__":
    main()