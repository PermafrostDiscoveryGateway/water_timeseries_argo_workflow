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

    # Check if NCO tools are installed
    logger.info("Checking for NCO tools...")
    try:
        subprocess.run(['ncks', '--version'], capture_output=True, check=True)
        logger.info("NCO tools found")
    except subprocess.CalledProcessError:
        logger.error("NCO tools not found. Please install: conda install -c conda-forge nco")
        return None
    except FileNotFoundError:
        logger.error("NCO tools not found. Please install: conda install -c conda-forge nco")
        return None

    # Find latest valid file
    logger.info("Finding most recent existing file...")
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
    logger.info(f"File size: {os.path.getsize(most_recent_dynamic_world_file) / (1024 ** 3):.2f} GB")

    # Get existing dates to check what's new
    logger.info("Getting existing dates...")
    try:
        # Use ncdump to get date values
        result = subprocess.run(
            ['ncdump', '-v', 'date', most_recent_dynamic_world_file],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse the output to get date values
        date_lines = [line for line in result.stdout.split('\n') if 'date =' in line or '}' in line]
        existing_dates = set()
        for line in date_lines:
            if 'date =' in line:
                parts = line.split('=')[1].strip().strip(';').split(',')
                for part in parts:
                    try:
                        existing_dates.add(int(part.strip()))
                    except ValueError:
                        pass
        logger.info(f"Found {len(existing_dates)} existing dates")
    except Exception as e:
        logger.warning(f"Could not parse existing dates: {e}")
        existing_dates = set()

    # Get new chunk files
    downloaded_files = glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc"))
    logger.info(f"Found {len(downloaded_files)} chunk files")

    if not downloaded_files:
        logger.error("No chunk files found!")
        return None

    # Create working directory for temp files
    temp_dir = "/tmp/netcdf_merge"
    os.makedirs(temp_dir, exist_ok=True)

    # Copy existing file as starting point
    working_file = os.path.join(temp_dir, "combined.nc")
    logger.info(f"Creating working copy: {working_file}")
    shutil.copy2(most_recent_dynamic_world_file, working_file)

    # Process each chunk
    days_offset = 3653
    ref_date = datetime(2015, 7, 1)
    all_new_dates = set()

    logger.info(f"Processing {len(downloaded_files)} chunk files...")

    for i, chunk_file in enumerate(downloaded_files):
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{len(downloaded_files)} chunks processed")

        try:
            # Create a temporary file with adjusted dates
            temp_chunk = os.path.join(temp_dir, f"temp_chunk_{i:04d}.nc")

            # Step 1: Adjust dates by adding offset
            logger.debug(f"  Adjusting dates in {os.path.basename(chunk_file)}")
            cmd = [
                'ncap2',
                '-s', f'date=date+{days_offset}',
                chunk_file,
                temp_chunk
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            # Step 2: Get dates in this chunk to check if they're new
            result = subprocess.run(
                ['ncdump', '-v', 'date', temp_chunk],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse dates
            chunk_dates = set()
            date_lines = [line for line in result.stdout.split('\n') if 'date =' in line or '}' in line]
            for line in date_lines:
                if 'date =' in line:
                    parts = line.split('=')[1].strip().strip(';').split(',')
                    for part in parts:
                        try:
                            chunk_dates.add(int(part.strip()))
                        except ValueError:
                            pass

            # Check if any dates are new
            new_dates_in_chunk = chunk_dates - existing_dates
            if new_dates_in_chunk:
                logger.debug(f"  Found new dates: {sorted(new_dates_in_chunk)}")
                all_new_dates.update(new_dates_in_chunk)

                # Step 3: Append to working file (only if there are new dates)
                logger.debug(f"  Appending to working file")
                cmd = ['ncks', '-A', temp_chunk, working_file]
                subprocess.run(cmd, capture_output=True, check=True)
            else:
                logger.debug(f"  No new dates in this chunk")

            # Clean up temp file
            if os.path.exists(temp_chunk):
                os.remove(temp_chunk)

        except subprocess.CalledProcessError as e:
            logger.error(f"Error processing {os.path.basename(chunk_file)}: {e}")
            if e.stderr:
                logger.error(f"  stderr: {e.stderr}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error processing {os.path.basename(chunk_file)}: {e}")
            continue

    if not all_new_dates:
        logger.warning("No new dates found in any chunk!")
        os.remove(working_file)
        shutil.rmtree(temp_dir)
        return None

    # Get latest date for filename
    latest_date = max(all_new_dates)
    latest_date_obj = ref_date + timedelta(days=int(latest_date))
    latest_date_string = latest_date_obj.strftime('%Y_%m_%d')
    logger.info(f"Latest date added: {latest_date_string}")

    # Final output filename
    new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    # Move working file to final location
    logger.info(f"Saving final file: {new_dynamic_world_data_file}")
    shutil.move(working_file, new_dynamic_world_data_file)

    # Clean up temp directory
    shutil.rmtree(temp_dir)

    # Verify the output
    logger.info("Verifying output...")
    try:
        result = subprocess.run(
            ['ncdump', '-h', new_dynamic_world_data_file],
            capture_output=True,
            text=True
        )
        logger.info("Output file verified")

        # Get final stats
        result = subprocess.run(
            ['ncdump', '-v', 'date', new_dynamic_world_data_file],
            capture_output=True,
            text=True
        )
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
        logger.info(f"  Date range: {min(final_dates)} to {max(final_dates)}")
        logger.info(f"  New dates added: {len(all_new_dates)}")
        logger.info("=" * 60)

    except Exception as e:
        logger.warning(f"Could not verify output: {e}")

    return new_dynamic_world_data_file


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()