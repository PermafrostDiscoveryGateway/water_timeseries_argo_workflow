from near_real_time_grid_v2 import verify_downloads_complete, verify_process_complete, merge_near_real_time_region, \
    process_near_real_time_region_dates_zarr, download_near_real_time_region_dates, generate_expected_dates, \
    merge_near_real_time_region_v3_simple, \
    compare_netcdf_files, verify_merged_netcdf, verify_merged_data, merge_near_real_time_region_v3_smart, \
    enable_memory_tracking, log_memory_usage, has_region_been_merged_for_dates, merge_near_real_time_region_v4_smart
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

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import xarray as xr
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
from loguru import logger
import tempfile


def human_readable_size(size_bytes: int, decimals: int = 2) -> str:
    """Convert bytes to human readable format."""
    if size_bytes == 0:
        return "0 B"

    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
    k = 1024
    index = 0

    while size_bytes >= k and index < len(units) - 1:
        size_bytes /= k
        index += 1

    return f"{size_bytes:.{decimals}f} {units[index]}"


def copy_netcdf_with_compression(
        source_path: str,
        target_path: str,
        chunk_size_ids: int = 5000,
        compression_level: int = 4,
        shuffle: bool = True
) -> bool:
    """
    Copy a NetCDF file to a new location with compression applied.

    Reads the source file in chunks to avoid loading everything into memory.
    Uses atomic write pattern to prevent corruption.
    """
    logger.info(f"Copying {source_path} to {target_path} with compression")

    # Get file size for progress
    source_size = Path(source_path).stat().st_size
    logger.info(f"Source file size: {human_readable_size(source_size)}")

    # Check if target already exists and is complete
    target_path = Path(target_path)
    if target_path.exists() and is_atomic_write_complete(target_path):
        logger.info(f"✅ Target file already exists and is complete: {target_path}")
        return True

    try:
        # Open source with chunking
        logger.info("Opening source file with chunking...")
        ds_source = xr.open_dataset(
            source_path,
            chunks={'id_geohash': chunk_size_ids, 'date': -1}
        )

        # Get dimensions
        total_ids = len(ds_source['id_geohash'])
        total_dates = len(ds_source['date'])
        logger.info(f"Source has {total_ids:,} IDs and {total_dates} dates")

        # Get all variable names
        var_names = list(ds_source.data_vars)
        logger.info(f"Variables to preserve: {var_names}")

        # Calculate number of chunks
        num_chunks = (total_ids + chunk_size_ids - 1) // chunk_size_ids
        logger.info(f"Processing {num_chunks} chunks of {chunk_size_ids} IDs each")

        # Process chunks
        chunk_file_paths = []
        temp_dir = tempfile.mkdtemp(prefix='copy_chunks_')

        try:
            for chunk_idx in tqdm(range(num_chunks), desc="Copying chunks"):
                start_idx = chunk_idx * chunk_size_ids
                end_idx = min((chunk_idx + 1) * chunk_size_ids, total_ids)

                logger.debug(f"Processing chunk {chunk_idx + 1}/{num_chunks} (IDs {start_idx:,} - {end_idx:,})")

                # Get chunk
                chunk_data = ds_source.isel(id_geohash=slice(start_idx, end_idx))

                # If chunk is empty, skip
                if len(chunk_data['id_geohash']) == 0:
                    continue

                # Write chunk to temporary file with compression
                chunk_file = Path(temp_dir) / f"chunk_{chunk_idx:04d}.nc"

                # Prepare encoding with compression
                encoding = {}
                for var in var_names:
                    if var in chunk_data.data_vars:
                        encoding[var] = {
                            'zlib': True,
                            'complevel': compression_level,
                            'shuffle': shuffle,
                            'chunksizes': (min(100, len(chunk_data['date'])), min(1000, len(chunk_data['id_geohash'])))
                        }

                # Write chunk
                chunk_data.to_netcdf(
                    chunk_file,
                    mode='w',
                    encoding=encoding,
                    unlimited_dims=['id_geohash']
                )

                chunk_file_paths.append(chunk_file)

                # Clean up
                chunk_data.close()
                gc.collect()

                logger.debug(
                    f"  Chunk {chunk_idx + 1} written: {chunk_file} ({human_readable_size(chunk_file.stat().st_size)})")

            # Close source
            ds_source.close()

            # Combine all chunks into final file
            logger.info(f"Combining {len(chunk_file_paths)} chunks into final file...")

            if len(chunk_file_paths) == 0:
                logger.error("No chunks were created")
                return False

            # Write to temp file first (atomic pattern)
            temp_final = target_path.parent / f".tmp_{target_path.name}_{int(time.time())}"

            if len(chunk_file_paths) == 1:
                # Only one chunk, just move it to temp
                shutil.move(chunk_file_paths[0], temp_final)
            else:
                # Combine multiple chunks to temp
                success = combine_chunks_to_netcdf(
                    chunk_files=chunk_file_paths,
                    output_path=temp_final,
                    compression_level=compression_level,
                    shuffle=shuffle
                )
                if not success:
                    return False

            # Verify temp file
            if not temp_final.exists() or temp_final.stat().st_size == 0:
                logger.error(f"Temp file is empty or missing: {temp_final}")
                return False

            # Verify temp file is valid
            try:
                verify_ds = xr.open_dataset(temp_final)
                verify_ds.close()
            except Exception as e:
                logger.error(f"Temp file verification failed: {e}")
                temp_final.unlink(missing_ok=True)
                return False

            # Atomic move to target
            if target_path.exists():
                backup_file = target_path.parent / f"{target_path.stem}_backup_{int(time.time())}{target_path.suffix}"
                shutil.move(str(target_path), str(backup_file))
                logger.info(f"Backed up existing file to: {backup_file}")

            shutil.move(str(temp_final), str(target_path))

            # Write completion marker
            write_completion_marker(target_path)

            # Clean up old backups (keep last 5)
            backup_files = sorted(target_path.parent.glob(f"{target_path.stem}_backup_*.nc"))
            for backup in backup_files[:-5]:
                try:
                    backup.unlink()
                    logger.debug(f"Removed old backup: {backup}")
                except:
                    pass

            final_size = target_path.stat().st_size
            compression_ratio = source_size / final_size if final_size > 0 else 0
            logger.info(f"✅ Copy completed successfully!")
            logger.info(f"  Original size: {human_readable_size(source_size)}")
            logger.info(f"  Compressed size: {human_readable_size(final_size)}")
            logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
            return True

        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        logger.error(f"Error copying with compression: {e}")
        import traceback
        traceback.print_exc()
        return False


def combine_chunks_to_netcdf(
        chunk_files: list,
        output_path: str,
        compression_level: int = 4,
        shuffle: bool = True
) -> bool:
    """
    Combine multiple chunk files into a single NetCDF file with compression.
    """
    logger.info(f"Combining {len(chunk_files)} chunks into {output_path}")

    try:
        # Open all chunks
        datasets = []
        for chunk_file in chunk_files:
            try:
                ds = xr.open_dataset(chunk_file)
                if len(ds['id_geohash']) > 0:
                    datasets.append(ds)
                else:
                    ds.close()
            except Exception as e:
                logger.warning(f"Could not open {chunk_file}: {e}")

        if not datasets:
            logger.error("No valid datasets to combine")
            return False

        # Concatenate all chunks
        logger.info(f"Concatenating {len(datasets)} datasets...")
        combined = xr.concat(datasets, dim='id_geohash')

        # Remove duplicate IDs if any
        _, unique_idx = np.unique(combined['id_geohash'].values, return_index=True)
        if len(unique_idx) < len(combined['id_geohash']):
            logger.info(f"Removing {len(combined['id_geohash']) - len(unique_idx)} duplicate IDs")
            combined = combined.isel(id_geohash=np.sort(unique_idx))

        # Sort by date and id
        combined = combined.sortby(['date', 'id_geohash'])

        # Prepare encoding with compression
        encoding = {}
        for var in combined.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': compression_level,
                'shuffle': shuffle,
                'chunksizes': (min(100, len(combined['date'])), min(1000, len(combined['id_geohash'])))
            }

        # Write to final file
        logger.info(f"Writing final file: {output_path}")
        combined.to_netcdf(
            output_path,
            mode='w',
            encoding=encoding,
            unlimited_dims=['id_geohash']
        )

        # Clean up
        combined.close()
        for ds in datasets:
            ds.close()
        gc.collect()

        logger.info(f"✅ Combined {len(datasets)} chunks into {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error combining chunks: {e}")
        import traceback
        traceback.print_exc()
        return False


def write_completion_marker(file_path: Path) -> None:
    """Write a marker file to indicate atomic write completion."""
    marker_file = file_path.parent / f".{file_path.name}.complete"
    with open(marker_file, 'w') as f:
        f.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"File size: {human_readable_size(file_path.stat().st_size)}\n")
        try:
            ds = xr.open_dataset(file_path)
            f.write(f"IDs: {len(ds['id_geohash']):,}\n")
            f.write(f"Dates: {len(ds['date'])}\n")
            ds.close()
        except:
            pass


def is_atomic_write_complete(file_path: str) -> bool:
    """Check if an atomic write was completed successfully."""
    file_path = Path(file_path)
    marker_file = file_path.parent / f".{file_path.name}.complete"

    if not file_path.exists():
        return False

    if not marker_file.exists():
        return False

    # Also verify the file can be opened
    try:
        ds = xr.open_dataset(file_path)
        ds.close()
        return True
    except:
        return False


def get_creation_time(filepath):
    """Get file creation time on Linux (birth time) if available"""
    stat_info = os.stat(Path(filepath))
    try:
        creation_time = stat_info.st_birthtime
    except AttributeError:
        creation_time = stat_info.st_ctime
    return creation_time


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


def get_complete_target_file(target_path: str) -> str:
    """
    Get the target file, recovering from backup if needed.
    Returns the path to the complete file, or None if unrecoverable.
    """
    target_path = Path(target_path)

    # If target doesn't exist, nothing to recover
    if not target_path.exists():
        return None

    # If target is complete, use it
    if is_atomic_write_complete(target_path):
        logger.info(f"✅ Target file is complete: {target_path}")
        return str(target_path)

    # Target exists but is incomplete - try to recover
    logger.warning(f"Target file exists but is incomplete: {target_path}")

    # Try to recover from backup
    backup_files = sorted(target_path.parent.glob(f"{target_path.stem}_backup_*.nc"))

    if backup_files:
        # Check each backup from newest to oldest
        for backup in reversed(backup_files):
            if is_atomic_write_complete(backup):
                logger.info(f"✅ Recovered from backup: {backup}")
                # Copy backup to target
                shutil.copy2(backup, target_path)
                # Write completion marker
                write_completion_marker(target_path)
                return str(target_path)

        # No valid backup found
        logger.error("No valid backup found. File may be corrupted.")
        return None
    else:
        logger.error("No backup found. File may be corrupted.")
        return None


def main():
    logger.debug(f"Beginning historical run")

    # Load environment
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        env_path = None
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

    REGION = os.environ.get("region_name", "TEST")
    SHOULD_RUN = False

    summer_months = [6, 7, 8, 9]
    dynamic_world_data_dir = os.environ['dynamic_world_data']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))

    # Find the most recent file
    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=lambda f: Path(f).stat().st_mtime)
    logger.debug(f"Most recent dynamic world file {most_recent_dynamic_world_file}")

    # Check if we should run
    TODAY = datetime.now()
    TODAY_MONTH = TODAY.month
    if TODAY_MONTH - 1 in summer_months:
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 3:
            SHOULD_RUN = True
            logger.debug(f"TODAY_DAY: {TODAY_DAY} should we run and check: {SHOULD_RUN}")

    if not SHOULD_RUN:
        logger.debug("Not time to run yet (need to be after 3rd of the month)")
        return

    # Get the base historical file
    # Try to find the original file (not historical_data_*)
    original_candidates = [f for f in all_dynamic_world_files if 'historical_data_' not in Path(f).name]
    if original_candidates:
        HISTORICAL_DATA_FILE = max(original_candidates, key=lambda f: Path(f).stat().st_mtime)
    else:
        HISTORICAL_DATA_FILE = most_recent_dynamic_world_file

    logger.info(f"Base historical file: {HISTORICAL_DATA_FILE}")
    logger.info(f"File size: {human_readable_size(Path(HISTORICAL_DATA_FILE).stat().st_size)}")

    # Determine target file name
    date_to_run = [datetime(TODAY.year, TODAY_MONTH - 1, 1).strftime("%Y-%m")]
    dates_to_run_string = date_to_run[0].replace('-', '_')
    NAME_OF_FINAL_MERGE_FILE = os.path.join(dynamic_world_data_dir, f"lakes_dw_Vdc{dates_to_run_string}.nc")
    logger.info(f"Target merge file: {NAME_OF_FINAL_MERGE_FILE}")

    # ========== CHECK IF TARGET FILE EXISTS AND IS COMPLETE ==========
    target_exists = Path(NAME_OF_FINAL_MERGE_FILE).exists()

    if target_exists:
        # Check if complete, try to recover if not
        complete_file = get_complete_target_file(NAME_OF_FINAL_MERGE_FILE)
        if complete_file:
            logger.info(f"✅ Using existing complete file: {complete_file}")
            # Update NAME_OF_FINAL_MERGE_FILE to the complete file path
            NAME_OF_FINAL_MERGE_FILE = complete_file
        else:
            logger.warning("Target file is corrupted and cannot be recovered. Will recreate.")
            # Remove corrupted file
            Path(NAME_OF_FINAL_MERGE_FILE).unlink(missing_ok=True)
            # Remove marker file
            marker_file = Path(NAME_OF_FINAL_MERGE_FILE).parent / f".{Path(NAME_OF_FINAL_MERGE_FILE).name}.complete"
            marker_file.unlink(missing_ok=True)
            # Copy with compression
            logger.info("Creating new compressed file...")
            copy_success = copy_netcdf_with_compression(
                source_path=HISTORICAL_DATA_FILE,
                target_path=NAME_OF_FINAL_MERGE_FILE,
                chunk_size_ids=5000,
                compression_level=4,
                shuffle=True
            )
            if not copy_success:
                logger.error("Failed to create new file")
                return
    else:
        # Target doesn't exist - create it with compression
        logger.info("Target file does not exist. Creating new compressed file...")
        copy_success = copy_netcdf_with_compression(
            source_path=HISTORICAL_DATA_FILE,
            target_path=NAME_OF_FINAL_MERGE_FILE,
            chunk_size_ids=5000,
            compression_level=4,
            shuffle=True
        )
        if not copy_success:
            logger.error("Failed to create new file")
            return

    logger.info(f"✅ Using target file: {NAME_OF_FINAL_MERGE_FILE}")
    logger.info(f"File size: {human_readable_size(Path(NAME_OF_FINAL_MERGE_FILE).stat().st_size)}")

    # ========== CHECK DOWNLOADS ==========
    logger.debug(f"Checking if we should merge for {date_to_run}")
    REGIONS = utils.region_boundaries.get_region_boundaries()
    REGION_NAMES = list(REGIONS.keys())

    regions_downloaded = 0
    regions_downloaded_names = []

    for region in REGION_NAMES:
        downloads_complete = verify_downloads_complete(
            region=region,
            analysis_dates=date_to_run,
            env_path=env_path
        )
        logger.debug(f"Downloads complete for {region}: {downloads_complete}")

        complete = downloads_complete['complete']
        summary = downloads_complete['summary']

        total_expected = summary['total_expected_downloads']
        total_successful = summary['total_successful_downloads']
        total_skipped = summary['total_skipped_downloads']
        total_available = total_successful + total_skipped

        if total_expected > 0:
            percent_downloaded = float(total_available) / float(total_expected)
            logger.debug(f"Percent downloaded for {region}: {percent_downloaded:.2%}")

            if downloads_complete['complete'] or percent_downloaded > 0.99:
                regions_downloaded += 1
                regions_downloaded_names.append(region)
        else:
            # No downloads expected for this region
            logger.info(f"Region {region} has no downloads expected")
            regions_downloaded += 1
            regions_downloaded_names.append(region)

    logger.info(f"Regions downloaded: {regions_downloaded}/{len(REGION_NAMES)}")
    logger.info(f"Regions: {regions_downloaded_names}")

    # ========== PERFORM MERGES ==========
    if regions_downloaded == len(REGION_NAMES):
        successfully_merged_region_count = 0
        failed_regions = []
        skipped_regions = []

        for region in REGION_NAMES:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"MERGING REGION: {region}")
            logger.info(f"{'=' * 60}")

            merge_result = merge_near_real_time_region_v4_smart(
                region=region,
                dates_to_merge=date_to_run,
                target_file_path=NAME_OF_FINAL_MERGE_FILE,
                env_path=env_path,
                skip_if_already_merged=True,
                verify_downloads_first=False  # Already verified
            )

            if merge_result.get('success', False):
                if merge_result.get('skipped', False):
                    logger.info(f"✅ Region {region}: SKIPPED (already complete)")
                    skipped_regions.append(region)
                else:
                    successfully_merged_region_count += 1
                    logger.info(f"✅ Region {region}: MERGED successfully")
                    logger.info(f"  IDs: {merge_result.get('id_count', 0):,}")
                    logger.info(f"  Size: {merge_result.get('file_size_gb', 0):.2f} GB")
            else:
                logger.error(f"❌ Region {region}: MERGE FAILED - {merge_result.get('error')}")
                failed_regions.append(region)

            # Show current file size
            if Path(NAME_OF_FINAL_MERGE_FILE).exists():
                logger.info(f"Current file size: {human_readable_size(Path(NAME_OF_FINAL_MERGE_FILE).stat().st_size)}")

            gc.collect()
            log_memory_usage(f"After merging region {region}")

        # ========== FINAL SUMMARY ==========
        logger.info(f"\n{'=' * 80}")
        logger.info("MERGE COMPLETION SUMMARY")
        logger.info(f"{'=' * 80}")
        logger.info(f"Total regions: {len(REGION_NAMES)}")
        logger.info(f"Successfully merged: {successfully_merged_region_count}")
        logger.info(f"Skipped (already complete): {len(skipped_regions)}")
        logger.info(f"Failed: {len(failed_regions)}")

        if skipped_regions:
            logger.info(f"Skipped regions: {skipped_regions}")
        if failed_regions:
            logger.error(f"Failed regions: {failed_regions}")

        if successfully_merged_region_count == len(REGION_NAMES):
            logger.info("✅ ALL regions merged successfully!")
            logger.info(f"Final file: {NAME_OF_FINAL_MERGE_FILE}")
            logger.info(f"Final size: {human_readable_size(Path(NAME_OF_FINAL_MERGE_FILE).stat().st_size)}")
        else:
            logger.warning(
                f"⚠️ Only {successfully_merged_region_count}/{len(REGION_NAMES)} regions merged successfully")
            logger.warning(f"Failed regions: {failed_regions}")
            logger.warning("Rerun the script to retry failed regions")
    else:
        logger.warning(f"Not all regions downloaded ({regions_downloaded}/{len(REGION_NAMES)})")
        logger.warning("Skipping merge until all downloads are complete")


if __name__ == "__main__":
    main()