import re
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
from loguru import logger
from google.cloud import storage
from google.auth import default

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _local_matches_remote(file_path: Path, blob) -> bool:
    """
    Determine whether the local file is already fully and correctly uploaded.

    Size alone can't detect a content change that happens to produce the same
    (or a smaller) file size, so we also compare local mtime/size against
    values recorded in the blob's custom metadata at upload time.
    """
    stat = file_path.stat()
    metadata = blob.metadata or {}
    stored_mtime = metadata.get('local_mtime')
    stored_size = metadata.get('local_size')

    if stored_mtime is None or stored_size is None:
        # No metadata recorded (e.g. blob predates this check) - fall back to
        # requiring an exact size match, which at least catches interrupted uploads.
        return stat.st_size == blob.size

    return str(stat.st_size) == stored_size and str(stat.st_mtime) == stored_mtime


def _upload_with_metadata(blob, file_path: Path):
    """Upload a file to GCS, stamping it with local mtime/size for future skip-checks."""
    stat = file_path.stat()
    blob.metadata = {
        'local_mtime': str(stat.st_mtime),
        'local_size': str(stat.st_size),
    }
    blob.upload_from_filename(str(file_path))


def get_storage_client():
    """Get authenticated storage client using application default credentials"""
    try:
        creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

        if creds_path and os.path.exists(creds_path):
            logger.info(f"Using credentials from: {creds_path}")
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
        return None


def sync_breakpoint_zarr_dirs_to_gcs(base_output_dir: str, bucket_name: str, path_to_cloud_folder: str,
                                     dry_run: bool = False):
    """
    Sync only breakpoint_zarr directories from local to GCS

    Args:
        base_output_dir: Base output directory (e.g., '/data/water_timeseries/output')
        bucket_name: GCS bucket name
        path_to_cloud_folder: Cloud folder path (e.g., 'water-timeseries-v2/data/output')
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
            return False

        logger.info(f"Connected to bucket: {bucket_name}")

        # Walk through local directory and find breakpoint_zarr directories
        base_dir = Path(base_output_dir)
        if not base_dir.exists():
            logger.error(f"Base output directory {base_output_dir} does not exist")
            return False

        # Find all breakpoint_zarr directories
        breakpoint_dirs = []
        for path in base_dir.rglob('*'):
            if path.is_dir() and path.name == 'breakpoint_zarr':
                breakpoint_dirs.append(path)

        if not breakpoint_dirs:
            logger.warning("No breakpoint_zarr directories found")
            return True

        logger.info(f"Found {len(breakpoint_dirs)} breakpoint_zarr directories to sync")

        # Process each breakpoint_zarr directory
        uploaded_count = 0
        skipped_count = 0
        error_count = 0

        for local_zarr_dir in breakpoint_dirs:
            # Get the relative path from base_dir
            rel_path = local_zarr_dir.relative_to(base_dir)
            # This will be something like 'ALASKA/breakpoint_zarr' or 'CANADA/breakpoint_zarr'
            cloud_dir = f"{path_to_cloud_folder}/{rel_path}"

            logger.info(f"Syncing: {local_zarr_dir} -> gs://{bucket_name}/{cloud_dir}")

            # Walk through files in this breakpoint_zarr directory
            for file_path in local_zarr_dir.rglob('*'):
                if not file_path.is_file():
                    continue

                # Get relative path from the breakpoint_zarr directory
                file_rel_path = file_path.relative_to(local_zarr_dir)
                blob_path = f"{cloud_dir}/{file_rel_path}"

                if dry_run:
                    logger.info(f"[DRY RUN] Would upload: {file_rel_path}")
                    uploaded_count += 1
                    continue

                blob = bucket.blob(blob_path)

                # Check if file already exists and matches the local file's mtime/size
                # (a size-only check can't tell a content change from an unchanged file
                # when the new size happens to be <= the old size)
                if blob.exists():
                    blob.reload()
                    if _local_matches_remote(file_path, blob):
                        skipped_count += 1
                        continue

                try:
                    logger.info(f"Uploading: {file_rel_path}")
                    _upload_with_metadata(blob, file_path)
                    uploaded_count += 1
                except Exception as e:
                    logger.error(f"Failed to upload {file_rel_path}: {e}")
                    error_count += 1

        logger.info(f"Sync summary: Uploaded {uploaded_count}, Skipped {skipped_count}, Errors {error_count}")

        if dry_run:
            logger.info(f"[DRY RUN] Would have uploaded {uploaded_count} files")

        return error_count == 0

    except Exception as e:
        logger.error(f"Error during sync: {e}")
        return False


def sync_breakpoint_parquet_files_to_gcs(base_output_dir: str, bucket_name: str, path_to_cloud_folder: str,
                                          dry_run: bool = False):
    """
    Sync only the final drain_*.parquet backup files (found in breakpoint_<date> directories)
    from local to GCS

    Args:
        base_output_dir: Base output directory (e.g., '/data/water_timeseries/output')
        bucket_name: GCS bucket name
        path_to_cloud_folder: Cloud folder path (e.g., 'water-timeseries-v2/data/output')
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
            return False

        logger.info(f"Connected to bucket: {bucket_name}")

        # Walk through local directory and find drain_*.parquet files
        base_dir = Path(base_output_dir)
        if not base_dir.exists():
            logger.error(f"Base output directory {base_output_dir} does not exist")
            return False

        drain_monthly_pattern = re.compile(r'^drain_\d{4}-\d{2}\.parquet$')
        parquet_files = [
            path for path in base_dir.rglob('drain_*.parquet')
            if path.is_file()
            and drain_monthly_pattern.match(path.name)
            and 'partial' not in path.name
        ]

        if not parquet_files:
            logger.warning("No drain_*.parquet files found")
            return True

        logger.info(f"Found {len(parquet_files)} drain_*.parquet files to sync")

        uploaded_count = 0
        skipped_count = 0
        error_count = 0

        for local_file in parquet_files:
            rel_path = local_file.relative_to(base_dir)
            blob_path = f"{path_to_cloud_folder}/{rel_path}"

            if dry_run:
                logger.info(f"[DRY RUN] Would upload: {rel_path}")
                uploaded_count += 1
                continue

            blob = bucket.blob(blob_path)

            # Check if file already exists and matches the local file's mtime/size
            # (a size-only check can't tell a content change from an unchanged file
            # when the new size happens to be <= the old size)
            if blob.exists():
                blob.reload()
                if _local_matches_remote(local_file, blob):
                    skipped_count += 1
                    continue

            try:
                logger.info(f"Uploading: {rel_path}")
                _upload_with_metadata(blob, local_file)
                uploaded_count += 1
            except Exception as e:
                logger.error(f"Failed to upload {rel_path}: {e}")
                error_count += 1

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
    dry_run = os.environ.get('dry_run', 'False').lower() in ('true', '1', 'yes')

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

    # Sync only breakpoint_zarr directories to cloud
    logger.info(f"Syncing breakpoint_zarr directories from {output_dir} to gs://{bucket_name}/{path_to_cloud_folder}")

    zarr_success = sync_breakpoint_zarr_dirs_to_gcs(
        base_output_dir=output_dir,
        bucket_name=bucket_name,
        path_to_cloud_folder=path_to_cloud_folder,
        dry_run=dry_run  # Set to True to preview changes
    )

    # Also sync the final parquet backup (drain_<date>.parquet) files
    logger.info(f"Syncing final drain_*.parquet files from {output_dir} to gs://{bucket_name}/{path_to_cloud_folder}")

    parquet_success = sync_breakpoint_parquet_files_to_gcs(
        base_output_dir=output_dir,
        bucket_name=bucket_name,
        path_to_cloud_folder=path_to_cloud_folder,
        dry_run=dry_run  # Set to True to preview changes
    )

    if zarr_success and parquet_success:
        logger.info("Sync complete!")
    else:
        logger.error("Sync failed!")


if __name__ == "__main__":
    main()