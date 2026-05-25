import subprocess
import os
import glob
import shutil
import sys
from datetime import datetime, timedelta
from loguru import logger
from dotenv import load_dotenv


def check_file_valid(filepath):
    """Check if a netCDF file is valid"""
    result = subprocess.run(['ncdump', '-h', filepath], capture_output=True, text=True)
    return result.returncode == 0


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
    logger.info(f"Most recent existing file: {os.path.basename(most_recent_dynamic_world_file)}")

    # Create temp directory
    temp_dir = "/tmp/fast_merge"
    os.makedirs(temp_dir, exist_ok=True)

    # Copy existing file as starting point
    working_file = os.path.join(temp_dir, "combined.nc")
    backup_file = os.path.join(temp_dir, "combined_backup.nc")

    logger.info(f"Creating working copy: {working_file}")
    shutil.copy2(most_recent_dynamic_world_file, working_file)
    shutil.copy2(most_recent_dynamic_world_file, backup_file)  # Keep a backup

    # Get chunk files
    chunk_files = sorted(glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc")))
    logger.info(f"Found {len(chunk_files)} chunk files to process")

    days_offset = 3653
    ref_date = datetime(2015, 7, 1)
    all_new_dates = set()
    processed = 0
    failed = 0
    failed_files = []

    logger.info(f"Starting to process {len(chunk_files)} files...")

    for i, chunk_file in enumerate(chunk_files):
        filename = os.path.basename(chunk_file)
        logger.info(f"Processing {i + 1}/{len(chunk_files)}: {filename}")

        # Verify working file is still valid before each append
        if not check_file_valid(working_file):
            logger.warning(f"  Working file corrupted! Restoring from backup...")
            shutil.copy2(backup_file, working_file)
            if not check_file_valid(working_file):
                logger.error(f"  Backup also corrupted! Cannot continue.")
                break

        # Create temp file
        temp_adjusted = os.path.join(temp_dir, f"adjusted_{i:04d}.nc")

        # Step 1: Get original dates
        cmd = ['ncdump', '-v', 'date', chunk_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        original_dates = []
        for line in result.stdout.split('\n'):
            if 'date =' in line:
                parts = line.split('=')[1].strip().strip(';').split(',')
                for part in parts:
                    try:
                        original_dates.append(int(part.strip()))
                    except ValueError:
                        pass

        # Skip if no dates or all dates already exist? Actually just process anyway
        if not original_dates:
            logger.debug(f"  No dates found, skipping")
            continue

        logger.debug(f"  Original dates: {original_dates}")

        # Step 2: Adjust dates using ncap2
        cmd = ['ncap2', '-O', '-s', f'date=date+{days_offset}', chunk_file, temp_adjusted]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.warning(f"  FAILED to adjust dates: {result.stderr[:100]}")
            failed += 1
            failed_files.append((filename, f"ncap2 failed"))
            continue

        # Get adjusted dates
        cmd = ['ncdump', '-v', 'date', temp_adjusted]
        result = subprocess.run(cmd, capture_output=True, text=True)
        adjusted_dates = []
        for line in result.stdout.split('\n'):
            if 'date =' in line:
                parts = line.split('=')[1].strip().strip(';').split(',')
                for part in parts:
                    try:
                        adjusted_dates.append(int(part.strip()))
                    except ValueError:
                        pass
        logger.debug(f"  Adjusted dates: {adjusted_dates}")

        # Step 3: Append to working file using ncks (without -v flag which was wrong)
        # The correct syntax is just ncks -A file_to_append target_file
        cmd = ['ncks', '-A', temp_adjusted, working_file]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"  FAILED to append: {result.stderr[:200]}")
            failed += 1
            failed_files.append((filename, f"ncks append failed"))

            # Try to recover by restoring backup and retrying this file?
            logger.info(f"  Attempting to recover working file from backup...")
            shutil.copy2(backup_file, working_file)

            # Retry this append once
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"  Recovery failed, skipping this file")
                continue
            else:
                logger.info(f"  Recovery successful!")
        else:
            logger.info(f"  SUCCESS - added dates: {adjusted_dates}")
            processed += 1
            for date_val in adjusted_dates:
                all_new_dates.add(date_val)

            # Update backup after successful append
            shutil.copy2(working_file, backup_file)

        # Clean up temp file
        if os.path.exists(temp_adjusted):
            os.remove(temp_adjusted)

        sys.stdout.flush()

    # Report summary
    logger.info("=" * 60)
    logger.info(f"Processing complete!")
    logger.info(f"  Successfully processed: {processed}")
    logger.info(f"  Failed: {failed}")

    if failed_files:
        logger.warning(f"Failed files ({len(failed_files)} total):")
        for fname, reason in failed_files[:10]:
            logger.warning(f"  - {fname}: {reason}")

    if not all_new_dates:
        logger.error("No dates were successfully added!")
        return None

    # Get latest date for filename
    latest_date = max(all_new_dates)
    latest_date_obj = ref_date + timedelta(days=int(latest_date))
    latest_date_string = latest_date_obj.strftime('%Y_%m_%d')

    # Final output
    new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    logger.info(f"Sorting dates in final file...")

    # Sort the final file by date
    sorted_file = os.path.join(temp_dir, "sorted.nc")

    # Try different sort methods
    cmd = ['ncks', '-O', '--msa', working_file, sorted_file]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.warning(f"First sort attempt failed: {result.stderr[:100]}")
        logger.info("Trying alternative sort method...")

        # Alternative: Use ncap2 to sort
        cmd = ['ncap2', '-O', '-s', "date[$date]=date", working_file, sorted_file]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Alternative sort also failed")
            # Just use unsorted file
            sorted_file = working_file

    logger.info(f"Saving final file: {new_dynamic_world_data_file}")
    shutil.move(sorted_file, new_dynamic_world_data_file)

    # Verify
    result = subprocess.run(['ncdump', '-v', 'date', new_dynamic_world_data_file], capture_output=True, text=True)
    final_dates = []
    for line in result.stdout.split('\n'):
        if 'date =' in line:
            parts = line.split('=')[1].strip().strip(';').split(',')
            for part in parts:
                try:
                    final_dates.append(int(part.strip()))
                except ValueError:
                    pass

    logger.info("=" * 60)
    logger.info(f"✓ MERGE COMPLETE!")
    logger.info(f"  Output: {new_dynamic_world_data_file}")
    logger.info(f"  Total dates: {len(final_dates)}")
    logger.info(f"  New dates added: {len(all_new_dates)}")
    logger.info("=" * 60)

    # Cleanup temp dir (optional - keep for debugging)
    # shutil.rmtree(temp_dir)

    return new_dynamic_world_data_file


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()