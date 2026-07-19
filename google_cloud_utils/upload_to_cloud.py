import sys
from pathlib import Path
from dotenv import load_dotenv
import os
from loguru import logger
from google.cloud import storage
from google.auth import default
from google.oauth2 import credentials

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def get_storage_client():
    """Get authenticated storage client using application default credentials"""
    try:
        # Use the default credentials - this handles both service accounts and OAuth2 user credentials
        creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

        if creds_path and os.path.exists(creds_path):
            logger.info(f"Using credentials from: {creds_path}")

            # Use the default credentials loader
            # This will work with both service account keys and OAuth2 user credentials
            credentials, project = default()

            client = storage.Client(
                credentials=credentials,
                project=os.environ.get('EE_PROJECT', 'pdg-project-406720')
            )

            logger.info("✅ Storage client created successfully")
            return client
        else:
            logger.error(f"Credentials file not found at: {creds_path}")
            return None

    except Exception as e:
        logger.error(f"Failed to create storage client: {e}")
        logger.error("Make sure your credentials file is valid and has the right permissions")
        return None


def sync_directory_to_gcs(output_dir: str, bucket_name: str, path_to_cloud_folder: str,
                          delete_extra_files: bool = False, dry_run: bool = False):
    """
    Sync a local directory to GCS using the Python client library

    Args:
        output_dir: Local directory to sync from
        bucket_name: GCS bucket name
        path_to_cloud_folder: Cloud folder path (e.g., 'water-timeseries-v2/data/output')
        delete_extra_files: If True, delete files in cloud that don't exist locally
        dry_run: If True, show what would happen without actually syncing
    """
    client = get_storage_client()
    if not client:
        return False

    try:
        bucket = client.bucket(bucket_name)

        # Verify bucket exists and is accessible
        if not bucket.exists():
            logger.error(f"Bucket {bucket_name} does not exist or is not accessible")
            logger.error("Make sure the bucket name is correct and you have permissions")
            return False

        logger.info(f"Connected to bucket: {bucket_name}")

        # Walk through local directory and upload files
        local_dir = Path(output_dir)
        if not local_dir.exists():
            logger.error(f"Local directory {output_dir} does not exist")
            return False

        # Get list of existing files in cloud
        existing_blobs = {}
        if delete_extra_files:
            logger.info("Building list of existing cloud files...")
            for blob in bucket.list_blobs(prefix=path_to_cloud_folder):
                # Get the relative path from the folder prefix
                rel_path = blob.name[len(path_to_cloud_folder):].lstrip('/')
                if rel_path:  # Skip empty
                    existing_blobs[rel_path] = blob
            logger.info(f"Found {len(existing_blobs)} existing files in cloud")

        # Collect local files
        local_files = {}
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(local_dir))
                local_files[rel_path] = file_path

        logger.info(f"Found {len(local_files)} files in local directory")

        # Upload new/changed files
        uploaded_count = 0
        skipped_count = 0
        error_count = 0

        for rel_path, file_path in local_files.items():
            blob_path = f"{path_to_cloud_folder}/{rel_path}"

            if dry_run:
                logger.info(f"[DRY RUN] Would upload: {rel_path}")
                uploaded_count += 1
                continue

            blob = bucket.blob(blob_path)

            # Check if file already exists
            if blob.exists():
                skipped_count += 1
                continue

            try:
                # Upload the file
                logger.info(f"Uploading: {rel_path}")
                blob.upload_from_filename(str(file_path))
                uploaded_count += 1
            except Exception as e:
                logger.error(f"Failed to upload {rel_path}: {e}")
                error_count += 1

        # Handle delete_extra_files
        if delete_extra_files and not dry_run:
            deleted_count = 0
            for cloud_file in existing_blobs:
                if cloud_file not in local_files:
                    blob_path = f"{path_to_cloud_folder}/{cloud_file}"
                    blob = bucket.blob(blob_path)
                    blob.delete()
                    logger.info(f"Deleted from cloud: {cloud_file}")
                    deleted_count += 1
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} files from cloud")

        logger.info(f"Sync summary: Uploaded {uploaded_count}, Skipped {skipped_count}, Errors {error_count}")

        if dry_run:
            logger.info(f"[DRY RUN] Would have uploaded {uploaded_count} files")

        return error_count == 0

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

    # Set environment variables
    os.environ["EE_PROJECT"] = project
    os.environ["CLOUDSDK_CORE_PROJECT"] = project

    bucket_name = 'pdg-storage-default'
    path_to_cloud_folder = 'water-timeseries-v2/data/output'

    # Sync local output_dir to cloud
    logger.info(f"Syncing {output_dir} to gs://{bucket_name}/{path_to_cloud_folder}")

    success = sync_directory_to_gcs(
        output_dir=output_dir,
        bucket_name=bucket_name,
        path_to_cloud_folder=path_to_cloud_folder,
        delete_extra_files=False,  # Set to True if you want to delete cloud files not in local
        dry_run=False  # Set to True to preview changes
    )

    if success:
        logger.info("Sync complete!")
    else:
        logger.error("Sync failed!")


if __name__ == "__main__":
    main()