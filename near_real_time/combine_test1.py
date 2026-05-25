import subprocess
import os
import glob
import shutil
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

    # Process each chunk file - simple append without checking duplicates
    chunk_files = sorted(glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc")))
    logger.info(f"Processing {len(chunk_files)} chunk files...")

    days_offset = 3653
    ref_date = datetime(2015, 7, 1)
    all_new_dates = set()

    for i, chunk_file in enumerate(chunk_files):
        if (i + 1) % 20 == 0:
            logger.info(f"Progress: {i + 1}/{len(chunk_files)}")

        # Create temp file with adjusted dates
        temp_adjusted = os.path.join(temp_dir, f"adjusted_{i:04d}.nc")

        # Adjust dates using ncap2
        cmd = ['ncap2', '-O', '-s', f'date=date+{days_offset}', chunk_file, temp_adjusted]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.warning(f"Failed to adjust dates in {os.path.basename(chunk_file)}: {result.stderr[:100]}")
            continue

        # Append to working file using ncks
        cmd = ['ncks', '-A', temp_adjusted, working_file]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.warning(f"Failed to append {os.path.basename(chunk_file)}: {result.stderr[:100]}")
        else:
            # Extract dates from this chunk to track what we added
            result = subprocess.run(['ncdump', '-v', 'date', temp_adjusted], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'date =' in line:
                    parts = line.split('=')[1].strip().strip(';').split(',')
                    for part in parts:
                        try:
                            all_new_dates.add(int(part.strip()))
                        except ValueError:
                            pass

        # Clean up temp file
        os.remove(temp_adjusted)

    if not all_new_dates:
        logger.warning("No dates were added!")
        return None

    # Get latest date for filename
    latest_date = max(all_new_dates)
    latest_date_obj = ref_date + timedelta(days=int(latest_date))
    latest_date_string = latest_date_obj.strftime('%Y_%m_%d')

    # Final output
    new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    logger.info(f"Saving final file: {new_dynamic_world_data_file}")

    # Sort the final file by date
    sorted_file = os.path.join(temp_dir, "sorted.nc")
    cmd = ['ncks', '-O', '--msa', working_file, sorted_file]
    subprocess.run(cmd, capture_output=True, check=True)

    shutil.move(sorted_file, new_dynamic_world_data_file)

    # Cleanup
    shutil.rmtree(temp_dir)

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
    logger.info(f"  New dates added (approx): {len(all_new_dates)}")
    logger.info("=" * 60)

    return new_dynamic_world_data_file


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()