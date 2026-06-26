#!/usr/bin/env python3
"""
Upload script for Google Cloud Storage
Usage: python upload_to_cloud_bucket.py <local_path>
"""

import sys
import os
from datetime import datetime
from pathlib import Path
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse


def upload_to_gcs(local_path, bucket_name, cloud_base_path, max_workers=5):
    """
    Upload contents of local_path to GCS bucket at cloud_base_path/YYYYMMDD/
    """
    # Get current date in YYYYMMDD format
    date_folder = datetime.now().strftime('%Y%m%d')

    # Construct the full cloud path
    full_cloud_path = f"{cloud_base_path}/{date_folder}"

    print(f"Uploading to: gs://{bucket_name}/{full_cloud_path}/")

    # Initialize GCS client with explicit project
    try:
        # Hardcode the project ID
        project_id = "pdg-project-406720"
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        print(f"✓ Connected to GCS bucket: {bucket_name} (project: {project_id})")
    except Exception as e:
        print(f"✗ Failed to connect to GCS: {e}")
        return False

    local_path_obj = Path(local_path)

    if not local_path_obj.exists():
        print(f"✗ Error: Path '{local_path}' does not exist")
        return False

    # Collect all files to upload
    files_to_upload = []
    if local_path_obj.is_dir():
        for file_path in local_path_obj.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path_obj)
                blob_name = f"{full_cloud_path}/{relative_path}".replace('\\', '/')
                files_to_upload.append((file_path, blob_name))
    else:
        blob_name = f"{full_cloud_path}/{local_path_obj.name}"
        files_to_upload.append((local_path_obj, blob_name))

    print(f"Found {len(files_to_upload)} files to upload")
    print("-" * 50)

    # Upload files in parallel
    uploaded_count = 0
    failed_count = 0
    lock = threading.Lock()

    def upload_file(file_path, blob_name):
        try:
            blob = bucket.blob(blob_name)
            # Use resumable upload for large files
            blob.upload_from_filename(str(file_path))
            return True, str(file_path), blob_name
        except Exception as e:
            return False, str(file_path), str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(upload_file, file_path, blob_name): (file_path, blob_name)
            for file_path, blob_name in files_to_upload
        }

        for future in as_completed(futures):
            success, path, info = future.result()
            with lock:
                if success:
                    uploaded_count += 1
                    print(f"✓ Uploaded: {path} -> gs://{bucket_name}/{info}")
                else:
                    failed_count += 1
                    print(f"✗ Failed: {path} - {info}")

    # Print summary
    print(f"\nUpload Summary:")
    print(f"  Successful: {uploaded_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Location: gs://{bucket_name}/{full_cloud_path}/")

    return failed_count == 0


def main():
    parser = argparse.ArgumentParser(description='Upload files to Google Cloud Storage')
    parser.add_argument('local_path', help='Local path to upload (file or directory)')
    parser.add_argument('--bucket', default='pdg-storage-default', help='GCS bucket name')
    parser.add_argument('--base-path', default='water_timeseries_v2/test_data/test_output',
                        help='Base path in the bucket')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel uploads')
    parser.add_argument('--project', default='pdg-project-406720', help='GCP project ID')

    args = parser.parse_args()

    print(f"Starting upload...")
    print(f"  Local path: {args.local_path}")
    print(f"  GCS bucket: {args.bucket}")
    print(f"  Cloud base path: {args.base_path}")
    print(f"  Parallel workers: {args.workers}")
    print(f"  GCP Project: {args.project}")
    print("-" * 50)

    success = upload_to_gcs(args.local_path, args.bucket, args.base_path, args.workers)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()