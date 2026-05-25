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

    # Step 1: Combine all chunks (fast concatenation)
    logger.info("Step 1: Combining all chunk files...")
    chunk_files = glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc"))
    combined_chunks = os.path.join(temp_dir, "all_chunks.nc")

    # Use ncrcat for fast concatenation
    cmd = ['ncrcat', '-h'] + chunk_files + [combined_chunks]
    subprocess.run(cmd, capture_output=True, check=True)
    logger.info(f"  Combined {len(chunk_files)} chunks")

    # Step 2: Adjust dates once
    logger.info("Step 2: Adjusting dates...")
    adjusted_chunks = os.path.join(temp_dir, "all_chunks_adjusted.nc")
    cmd = ['ncap2', '-O', '-s', 'date=date+3653', combined_chunks, adjusted_chunks]
    subprocess.run(cmd, capture_output=True, check=True)

    # Step 3: Merge with existing file
    logger.info("Step 3: Merging with existing file...")
    merged_file = os.path.join(temp_dir, "merged.nc")

    # Use ncecat to concatenate along date dimension
    cmd = ['ncecat', '-O', '-u', 'date', most_recent_dynamic_world_file, adjusted_chunks, merged_file]
    subprocess.run(cmd, capture_output=True, check=True)

    # Step 4: Sort by date
    logger.info("Step 4: Sorting by date...")
    sorted_file = os.path.join(temp_dir, "sorted.nc")
    cmd = ['ncks', '-O', '--msa', merged_file, sorted_file]
    subprocess.run(cmd, capture_output=True, check=True)

    # Get latest date for filename
    result = subprocess.run(['ncdump', '-v', 'date', sorted_file], capture_output=True, text=True)
    dates = []
    for line in result.stdout.split('\n'):
        if 'date =' in line:
            parts = line.split('=')[1].strip().strip(';').split(',')
            for part in parts:
                try:
                    dates.append(int(part.strip()))
                except ValueError:
                    pass

    latest_date = max(dates)
    ref_date = datetime(2015, 7, 1)
    latest_date_obj = ref_date + timedelta(days=int(latest_date))
    latest_date_string = latest_date_obj.strftime('%Y_%m_%d')

    # Final output
    new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    logger.info(f"Step 5: Saving final file...")
    shutil.move(sorted_file, new_dynamic_world_data_file)

    # Cleanup
    shutil.rmtree(temp_dir)

    logger.info("=" * 60)
    logger.info(f"✓ MERGE COMPLETE!")
    logger.info(f"  Output: {new_dynamic_world_data_file}")
    logger.info(f"  Total dates: {len(dates)}")
    logger.info("=" * 60)

    return new_dynamic_world_data_file


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()