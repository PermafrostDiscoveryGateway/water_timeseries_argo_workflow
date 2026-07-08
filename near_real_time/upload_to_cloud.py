import sys
from pathlib import Path
from dotenv import load_dotenv
import time
import os
from loguru import logger
import subprocess
import shutil

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def check_gsutil_installed():
    """Check if gsutil is installed and available"""
    if shutil.which('gsutil') is None:
        logger.error("gsutil not found. Please install Google Cloud SDK")
        logger.error("Installation: https://cloud.google.com/sdk/docs/install")
        return False
    return True


def sync_with_gsutil(output_dir: str, bucket_name: str, path_to_cloud_folder: str,
                     delete_extra_files: bool = False, dry_run: bool = False):
    """
    Sync using gsutil rsync command

    Args:
        output_dir: Local directory to sync from
        bucket_name: GCS bucket name
        path_to_cloud_folder: Cloud folder path
        delete_extra_files: If True, delete files in cloud that don't exist locally
        dry_run: If True, show what would happen without actually syncing
    """
    if not check_gsutil_installed():
        return False

    # Build the gsutil command
    cloud_path = f"gs://{bucket_name}/{path_to_cloud_folder}"

    # Base command
    cmd = ['gsutil', 'rsync']

    # Add options
    if delete_extra_files:
        cmd.append('-d')  # Delete extra files in destination

    if dry_run:
        cmd.append('-n')  # Dry run (no actual changes)

    # Add options for better performance
    cmd.extend(['-r'])  # Recursive
    cmd.extend(['-e'])  # Skip symlinks

    # Add source and destination
    cmd.append(output_dir)
    cmd.append(cloud_path)

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            if dry_run:
                logger.info("Dry run completed. No changes made.")
                logger.info(result.stdout)
            else:
                logger.info("Sync completed successfully!")
                # Parse output to show what was synced
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if 'Uploading' in line or 'Copying' in line:
                            logger.debug(line)
        else:
            logger.error(f"Sync failed with code {result.returncode}")
            logger.error(f"Error: {result.stderr}")
            return False

        return True

    except subprocess.TimeoutExpired:
        logger.error("Sync timed out")
        return False
    except Exception as e:
        logger.error(f"Error during sync: {e}")
        return False


def main():
    logger.debug(f"Checking if we should upload anything to cloud")

    # Get environment variables
    output_dir = os.environ.get('output_dir')
    project = os.environ.get('project')

    if not output_dir:
        logger.error("output_dir environment variable not set")
        return

    if not project:
        logger.error("project environment variable not set")
        return

    # Verify output directory exists
    if not Path(output_dir).exists():
        logger.error(f"Output directory {output_dir} does not exist")
        return

    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID

    CLOUD_PROJECT = project
    bucket_name = 'pdg-storage-default'
    path_to_cloud_folder = 'water-timeseries-v2/data/output'

    # Set up authentication for gsutil
    # gsutil uses the same credentials as gcloud
    os.environ["CLOUDSDK_CORE_PROJECT"] = project

    # Sync local output_dir to cloud
    logger.info(f"Syncing {output_dir} to gs://{bucket_name}/{path_to_cloud_folder}")

    # Option 1: Normal sync - only uploads changed/new files
    success = sync_with_gsutil(
        output_dir,
        bucket_name,
        path_to_cloud_folder,
        delete_extra_files=False,  # Set to True if you want to delete cloud files not in local
        dry_run=False  # Set to True to preview changes
    )

    if success:
        logger.info("Sync complete!")
    else:
        logger.error("Sync failed!")


if __name__ == "__main__":
    main()