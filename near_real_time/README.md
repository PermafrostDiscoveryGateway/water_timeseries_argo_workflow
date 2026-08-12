## Instructions for Near Real Time 

Running the near real time pipeline involves running the following scripts, in order.

1. `download_region.py` . This downloads the most recent month's data for a region. It is run every month during the summer, plus October, since it will be downloading September's data.
2. Once the downloads are finished, the next step is `merge_recent_downloads.py` . The download scripts downloads a bunch of smaller .nc files for each bounding box. This script, for each region, merges them all into a single file per region.
3. After the merge is finished, then you can run `process_NRT.py`. This processes using the most recent historical files, and the new data for each region.
4. Data is then uploaded using the script in `google_cloud_utils/upload_to_cloud.py` . It is set in the cron jobs to check not just if files or directories exist, but if the file on disk is larger than the one currently in the bucket.
5. Then `create_new_historical_file.py` is run. This combines the new merged data from each region, with the previous latest historical file, to create a new one. For now, this is meant to happen after the process, because otherwise, if this file is being written, the process script might start running with a file still being written. 

### Testing