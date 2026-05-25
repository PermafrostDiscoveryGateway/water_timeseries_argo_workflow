import subprocess
import os
import glob
import shutil
import sys
from datetime import datetime, timedelta
from loguru import logger
from dotenv import load_dotenv


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
    logger.info(f"Creating working copy: {working_file}")
    shutil.copy2(most_recent_dynamic_world_file, working_file)

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

        # Create temp files
        temp_adjusted = os.path.join(temp_dir, f"adjusted_{i:04d}.nc")
        temp_debug = os.path.join(temp_dir, f"debug_{i:04d}.txt")

        # Step 1: Check original file structure
        logger.debug(f"  Checking original file...")
        cmd = ['ncdump', '-h', chunk_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"  File appears corrupted: {result.stderr[:200]}")
            failed += 1
            failed_files.append((filename, "corrupted file"))
            continue

        # Step 2: Get original dates
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
        logger.debug(f"  Original dates: {original_dates}")

        # Step 3: Adjust dates using ncap2 with full error output
        logger.debug(f"  Adjusting dates...")
        cmd = ['ncap2', '-O', '-v', '-s', f'date=date+{days_offset}', chunk_file, temp_adjusted]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"  FAILED to adjust dates:")
            logger.error(f"    Return code: {result.returncode}")
            logger.error(f"    Stdout: {result.stdout[:200]}")
            logger.error(f"    Stderr: {result.stderr[:500]}")
            failed += 1
            failed_files.append((filename, f"ncap2 failed: {result.stderr[:100]}"))

            # Save debug info
            with open(temp_debug, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"Stdout:\n{result.stdout}\n")
                f.write(f"Stderr:\n{result.stderr}\n")
            continue

        # Step 4: Check adjusted dates
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

        # Step 5: Append to working file with full error output
        logger.debug(f"  Appending to working file...")
        cmd = ['ncks', '-A', '-v', temp_adjusted, working_file]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"  FAILED to append:")
            logger.error(f"    Return code: {result.returncode}")
            logger.error(f"    Stdout: {result.stdout[:200]}")
            logger.error(f"    Stderr: {result.stderr[:500]}")
            failed += 1
            failed_files.append((filename, f"ncks append failed: {result.stderr[:100]}"))

            # Save debug info
            with open(temp_debug, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"Stdout:\n{result.stdout}\n")
                f.write(f"Stderr:\n{result.stderr}\n")
        else:
            logger.info(f"  SUCCESS - added dates: {adjusted_dates}")
            processed += 1
            for date_val in adjusted_dates:
                all_new_dates.add(date_val)

        # Clean up temp files
        if os.path.exists(temp_adjusted):
            os.remove(temp_adjusted)
        if os.path.exists(temp_debug):
            # Keep debug files for failed ones
            if result.returncode == 0:
                os.remove(temp_debug)

        sys.stdout.flush()

    # Report summary
    logger.info("=" * 60)
    logger.info(f"Processing complete!")
    logger.info(f"  Successfully processed: {processed}")
    logger.info(f"  Failed: {failed}")

    if failed_files:
        logger.warning(f"Failed files:")
        for fname, reason in failed_files[:10]:  # Show first 10
            logger.warning(f"  - {fname}: {reason}")

        # Save full list to file
        failed_log = os.path.join(temp_dir, "failed_files.txt")
        with open(failed_log, 'w') as f:
            for fname, reason in failed_files:
                f.write(f"{fname}\t{reason}\n")
        logger.info(f"Full failed list saved to: {failed_log}")

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
    cmd = ['ncks', '-O', '--msa', working_file, sorted_file]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to sort: {result.stderr}")
        # Try alternative sort method
        logger.info("Trying alternative sort method...")
        cmd = ['ncap2', '-O', '-s', 'date[$date]=date', working_file, sorted_file]
        subprocess.run(cmd, capture_output=True, check=True)

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

    return new_dynamic_world_data_file


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()