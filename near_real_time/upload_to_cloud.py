import sys
from pathlib import Path
from dotenv import load_dotenv
import os
from loguru import logger
import subprocess
import shutil
from google.cloud import storage
from google.oauth2 import service_account
from google.api_core import exceptions

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


def test_gcs_authentication():
    """Test GCS authentication before attempting sync"""
    try:
        creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if creds_path and os.path.exists(creds_path):
            logger.info(f"Testing authentication with credentials from: {creds_path}")

            # Try to create a storage client
            credentials = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            client = storage.Client(
                credentials=credentials,
                project=os.environ.get('EE_PROJECT', 'pdg-project-406720')
            )

            # Try to list the bucket (just get 1 blob to test)
            bucket = client.bucket('pdg-storage-default')
            blobs = list(bucket.list_blobs(max_results=1))
            logger.info("✅ Authentication successful!")
            return True
        else:
            logger.error(f"Credentials file not found: {creds_path}")
            return False
    except Exception as e:
        logger.error(f"❌ Authentication test failed: {e}")
        return False


def sync_with_gcs_client(output_dir: str, bucket_name: str, path_to_cloud_folder: str,
                         delete_extra_files: bool = False, dry_run: bool = False):
    """
    Sync using Google Cloud Storage client library
    """
    try:
        # Get credentials
        creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

        if creds_path and os.path.exists(creds_path):
            logger.info(f"Using credentials from: {creds_path}")
            credentials = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            client = storage.Client(
                credentials=credentials,
                project=os.environ.get('EE_PROJECT', 'pdg-project-406720')
            )
        else:
            logger.warning("No credentials file found, using default credentials")
            client = storage.Client(project=os.environ.get('EE_PROJECT', 'pdg-project-406720'))

        bucket = client.bucket(bucket_name)

        # Verify bucket exists
        if not bucket.exists():
            logger.error(f"Bucket {bucket_name} does not exist or is not accessible")
            return False

        logger.info(f"Connected to bucket: {bucket_name}")

        # Walk through local directory and upload files
        local_dir = Path(output_dir)
        if not local_dir.exists():
            logger.error(f"Local directory {output_dir} does not exist")
            return False

        uploaded_count = 0
        skipped_count = 0
        error_count = 0

        # Get list of existing blobs in cloud for comparison (if delete_extra_files)
        existing_blobs = set()
        if delete_extra_files:
            logger.info("Building list of existing cloud files...")
            for blob in bucket.list_blobs(prefix=path_to_cloud_folder):
                # Get the relative path from the folder prefix
                rel_path = blob.name[len(path_to_cloud_folder):].lstrip('/')
                existing_blobs.add(rel_path)
            logger.info(f"Found {len(existing_blobs)} existing files in cloud")

        # Upload local files
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                # Calculate relative path for cloud storage
                rel_path = str(file_path.relative_to(local_dir))
                blob_path = f"{path_to_cloud_folder}/{rel_path}"

                if dry_run:
                    logger.info(f"[DRY RUN] Would upload: {rel_path}")
                    uploaded_count += 1
                    continue

                blob = bucket.blob(blob_path)

                # Check if file already exists
                if blob.exists():
                    # Optionally compare content or metadata to decide if we should skip
                    # For now, just skip existing files
                    skipped_count += 1
                    continue

                try:
                    # Upload the file
                    blob.upload_from_filename(str(file_path))
                    uploaded_count += 1
                    logger.info(f"✅ Uploaded: {rel_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to upload {rel_path}: {e}")
                    error_count += 1

        # Handle delete_extra_files
        if delete_extra_files and not dry_run:
            # Collect all local file paths
            local_files = set()
            for file_path in local_dir.rglob('*'):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(local_dir))
                    local_files.add(rel_path)

            # Delete cloud files that don't exist locally
            for cloud_file in existing_blobs:
                if cloud_file not in local_files and cloud_file:  # Skip empty
                    blob_path = f"{path_to_cloud_folder}/{cloud_file}"
                    blob = bucket.blob(blob_path)
                    blob.delete()
                    logger.info(f"🗑️ Deleted from cloud: {cloud_file}")

        logger.info(
            f"📊 Sync summary: Uploaded {uploaded_count} files, Skipped {skipped_count} existing, Errors {error_count}")

        if dry_run:
            logger.info(f"[DRY RUN] Would have uploaded {uploaded_count} files")

        return error_count == 0

    except exceptions.PermissionDenied as e:
        logger.error(f"❌ Permission denied: {e}")
        logger.error("Make sure the service account has these permissions:")
        logger.error("  - storage.objects.list")
        logger.error("  - storage.objects.create")
        logger.error("  - storage.objects.delete (if using delete_extra_files)")
        return False
    except Exception as e:
        logger.error(f"❌ Error during sync: {e}")
        return False


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
    logger.debug("Checking if we should upload anything to cloud")

    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

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

    bucket_name = 'pdg-storage-default'
    path_to_cloud_folder = 'water-timeseries-v2/data/output'

    # Set up authentication
    os.environ["CLOUDSDK_CORE_PROJECT"] = project

    # Test authentication first
    logger.info("🔐 Testing authentication...")
    if not test_gcs_authentication():
        logger.error("Authentication failed. Please check your credentials.")
        return

    # Sync local output_dir to cloud
    logger.info(f"📤 Syncing {output_dir} to gs://{bucket_name}/{path_to_cloud_folder}")

    # Use the Python client library
    success = sync_with_gcs_client(
        output_dir,
        bucket_name,
        path_to_cloud_folder,
        delete_extra_files=False,  # Set to True if you want to delete cloud files not in local
        dry_run=False  # Set to True to preview changes
    )

    if success:
        logger.info("✅ Sync complete!")
    else:
        logger.error("❌ Sync failed!")


if __name__ == "__main__":
    main()