import netCDF4 as nc
import pandas as pd
from netCDF4 import num2date
from datetime import datetime
from loguru import logger
import os
import numpy as np
import glob
import xarray as xr
from .combine_chunks_with_previous import combine_new_dynamic_world_data_with_latest
from dotenv import load_dotenv
from water_timeseries.downloader import EarthEngineDownloader

def get_date_label(current_date):
    return str(current_date).split(' ')[0].replace('-', '_')


def find_missing_summer_dates(existing_dates, current_date=None):
    """
    Find missing June, July, August, September dates that are not in the existing list.

    Parameters:
    -----------
    existing_dates : list or pandas.DatetimeIndex
        List of dates that already exist in the file
    current_date : datetime or str, optional
        The date to check up to (defaults to current date)

    Returns:
    --------
    list of datetime objects for missing summer months (June-September)
    """

    # Convert existing_dates to pandas DatetimeIndex if not already
    if not isinstance(existing_dates, pd.DatetimeIndex):
        existing_dates = pd.to_datetime(pd.Series(existing_dates))

    # Set current date if not provided
    if current_date is None:
        current_date = datetime.now()
    elif isinstance(current_date, str):
        current_date = pd.to_datetime(current_date)

    # Get the earliest date from existing data
    earliest_date = existing_dates.min()

    # Generate all possible summer months (June-September) from earliest_date to current_date
    all_summer_dates = []

    # Start from the year of earliest_date
    start_year = earliest_date.year
    end_year = current_date.year
    end_month = current_date.month

    for year in range(start_year, end_year + 1):
        # Determine which summer months to include for each year
        summer_months = [6, 7, 8, 9]  # June, July, August, September

        # For the last year, only include months up to current month
        if year == end_year:
            summer_months = [m for m in summer_months if m <= end_month]

        # For the first year, only include months >= earliest date's month
        if year == start_year:
            start_month = earliest_date.month
            summer_months = [m for m in summer_months if m >= start_month]

        # Create date objects for each summer month (always on the 1st)
        for month in summer_months:
            date = pd.Timestamp(year=year, month=month, day=1)
            all_summer_dates.append(date)

    # Find which dates are missing (not in existing_dates)
    existing_set = set(existing_dates)
    missing_dates = [date for date in all_summer_dates if date not in existing_set]

    return missing_dates

def find_missing_summer_months(existing_dates, current_date=None):
    """
    Alternative function that returns a more readable summary of missing months.

    Returns:
    --------
    dict with years as keys and lists of missing months as values
    """

    missing_dates = find_missing_summer_dates(existing_dates, current_date)

    # Group missing dates by year
    missing_by_year = {}
    for date in missing_dates:
        year = date.year
        month = date.month
        if year not in missing_by_year:
            missing_by_year[year] = []
        missing_by_year[year].append(month)

    return missing_by_year

def get_all_dates_simple(netcdf_path):
    """
    Simple function to get all dates as pandas datetime objects.
    Confirms all dates are on the 1st of the month.
    """
    with nc.Dataset(netcdf_path, 'r') as ds:
        # Get the date values and units
        date_values = ds.variables['date'][:]
        units = ds.variables['date'].units
        calendar = getattr(ds.variables['date'], 'calendar', 'proleptic_gregorian')

        # Convert to cftime objects first
        cftime_dates = num2date(date_values, units=units, calendar=calendar)

        # Convert cftime to pandas datetime (method 1 - using xarray's helper)
        # Convert each cftime object to a pandas Timestamp
        dates_pd = pd.to_datetime([str(d) for d in cftime_dates])

        # Alternative method (more efficient if you have many dates):
        # dates_pd = pd.Series(cftime_dates).apply(lambda x: pd.Timestamp(x.year, x.month, x.day))

        # Verify they're all on the 1st
        if all(dates_pd.dt.day == 1):
            print("✓ All dates are on the 1st of the month")
        else:
            print("⚠ Warning: Not all dates are on the 1st of the month")

        return dates_pd

# Or even simpler - use xarray directly (recommended for your use case)
def get_dates_xarray(netcdf_path):
    """Get dates using xarray - simplest approach"""
    import xarray as xr
    ds = xr.open_dataset(netcdf_path)
    # xarray already converts cftime to datetime64
    dates = pd.to_datetime(ds.date.values)
    ds.close()
    return dates

# Or using pure netCDF4 with manual conversion (most robust)
def get_dates_manual(netcdf_path):
    """Manually parse dates from units - avoids cftime entirely"""
    with nc.Dataset(netcdf_path, 'r') as ds:
        date_values = ds.variables['date'][:]
        units = ds.variables['date'].units

        # Parse the reference date from units string (e.g., "days since 2015-07-01")
        import re
        match = re.search(r'days since (\d{4}-\d{2}-\d{2})', units)
        if match:
            base_date = pd.Timestamp(match.group(1))
            # Convert numeric days to actual dates
            dates = [base_date + pd.Timedelta(days=int(d)) for d in date_values]

            dates_pd = pd.Series(dates)

            # Verify they're all on the 1st
            if all(dates_pd.dt.day == 1):
                print("✓ All dates are on the 1st of the month")
            else:
                print("⚠ Warning: Not all dates are on the 1st of the month")

            return dates_pd
        else:
            raise ValueError(f"Could not parse units: {units}")

def check_missing_data_in_netcdf(netcdf_path):
    """
    Check which summer months are missing from the NetCDF file.
    """
    import xarray as xr
    import pandas as pd
    from datetime import datetime

    # Load dates from NetCDF
    ds = xr.open_dataset(netcdf_path)
    existing_dates = pd.to_datetime(ds.date.values)
    ds.close()

    print(f"Existing dates in file: {len(existing_dates)}")
    print(f"Date range: {existing_dates.min()} to {existing_dates.max()}")

    # Find missing dates up to present
    missing_dates = find_missing_summer_dates(existing_dates)

    if missing_dates:
        print(f"\n⚠ Missing {len(missing_dates)} summer months:")
        for date in missing_dates:
            print(f"  {date.strftime('%Y-%m-%d')}")

        # Check if current month is June-September and missing
        current = datetime.now()
        current_summer = current.month in [6, 7, 8, 9]
        current_exists = pd.Timestamp(current.year, current.month, 1) in existing_dates

        if current_summer and not current_exists:
            print(f"\n⚠ Current month ({current.strftime('%B %Y')}) is missing!")

        return missing_dates
    else:
        print("\n✓ All summer months are present!")
        return []


def download_new_dynamic_world_data(env_path=None):
    if env_path is None:
        load_dotenv()
        logger.info("Loading environment from default .env file")
    else:
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")

    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID
    dynamic_world_dir = os.environ['dynamic_world_dir']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
    dynamic_world_data_file = os.environ['dynamic_world_data_file']
    vector_lake_file = os.environ['vector_lake_file']
    new_dynamic_world_data_dir = os.environ['new_dynamic_world_data_dir']

    logger.debug(f"Dynamic world data file: {dynamic_world_data_file}")
    logger.debug(f"Vector lake file: {vector_lake_file}")
    logger.debug(f"New dynamic world data dir: {new_dynamic_world_data_dir}")

    missing_dates = check_missing_data_in_netcdf(most_recent_dynamic_world_file)

    if not missing_dates or len(missing_dates) == 0:
        logger.debug(f"No missing dates found in {dynamic_world_data_file}")
        return most_recent_dynamic_world_file
    logger.debug(f"Missing dates found are {missing_dates}")
    missing_years = []
    missing_months = []
    for date in missing_dates:
        if date.year not in missing_years:
            missing_years.append(date.year)
        if date.month not in missing_months:
            missing_months.append(date.month)

    logger.debug(f"Missing years: {missing_years}")
    logger.debug(f"Missing months: {missing_months}")

    logger.debug(f"Downloading new dynamic world data")

    current_date = str(datetime.now())
    current_date_stamp = current_date.split(' ')[0]
    current_date_stamp = current_date_stamp.replace('-', '_')
    download_filename = 'dynamic_world_download_' + current_date_stamp + '.nc'
    download_filepath = os.path.join(new_dynamic_world_data_dir, download_filename)
    # DOWNLOAD INDIVIDUAL NETCDF FILES FROM THE SPLIT FILES
    dl = EarthEngineDownloader(ee_auth=True, logger=logger)
    ds = dl.download_dw_monthly(
        vector_dataset=vector_lake_file,
        name_attribute="id_geohash",
        years=missing_years,
        months=missing_months,
        save_to_file=download_filepath,
        max_total_requests=500,
        n_parallel=1,
    )

    logger.debug(f"Finished downloading to {download_filepath}")

    # Merge the existing and new data
    logger.debug("Merging existing data with new data...")

    # Open both datasets using xarray
    existing_ds = xr.open_dataset(most_recent_dynamic_world_file)
    new_ds = xr.open_dataset(download_filepath)

    # Verify no date overlap (should be true based on missing dates logic)
    existing_dates = set(pd.to_datetime(existing_ds.date.values))
    new_dates = set(pd.to_datetime(new_ds.date.values))
    overlapping_dates = existing_dates & new_dates

    if overlapping_dates:
        logger.warning(f"Found {len(overlapping_dates)} overlapping dates: {sorted(overlapping_dates)}")
        logger.warning("Removing overlapping dates from new dataset...")
        # Remove overlapping dates from new dataset
        mask = ~new_ds.date.isin(list(overlapping_dates))
        new_ds = new_ds.sel(date=mask)

    # Simple concatenation along date dimension
    # This doesn't require matching id_geohash values
    merged_ds = xr.concat([existing_ds, new_ds], dim="date")

    # Sort by date
    merged_ds = merged_ds.sortby("date")

    # Verify the merge was successful
    final_dates = pd.to_datetime(merged_ds.date.values)
    logger.info(f"Original dates: {len(existing_dates)}")
    logger.info(f"New dates added: {len(new_dates)}")
    logger.info(f"Total dates after merge: {len(final_dates)}")
    logger.info(f"Date range after merge: {final_dates.min()} to {final_dates.max()}")

    # Create the final filename with timestamp
    new_dynamic_world_filename = 'lakes_dw_V2d_' + current_date_stamp + '.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    # Save the merged dataset
    merged_ds.to_netcdf(new_dynamic_world_data_file)

    logger.debug(f"Successfully merged datasets!")
    logger.debug(f"Original file: {most_recent_dynamic_world_file}")
    logger.debug(f"New data file: {download_filepath}")
    logger.debug(f"Merged file saved as: {new_dynamic_world_data_file}")

    # Optional: Print summary statistics about lake coverage
    existing_lakes = set(existing_ds.id_geohash.values)
    new_lakes = set(new_ds.id_geohash.values)
    common_lakes = existing_lakes & new_lakes
    only_in_existing = existing_lakes - new_lakes
    only_in_new = new_lakes - existing_lakes

    logger.info(f"Lake coverage summary:")
    logger.info(f"  - Lakes in existing dataset: {len(existing_lakes)}")
    logger.info(f"  - Lakes in new dataset: {len(new_lakes)}")
    logger.info(f"  - Common lakes: {len(common_lakes)}")
    logger.info(f"  - Lakes only in existing: {len(only_in_existing)}")
    logger.info(f"  - Lakes only in new: {len(only_in_new)}")

    # Close datasets to free memory
    existing_ds.close()
    new_ds.close()
    merged_ds.close()

    return new_dynamic_world_data_file

def download_new_dynamic_world_data_split_files_v1(env_path=None):
    """
    Download new Dynamic World data for missing dates using split vector files.

    Parameters:
    -----------
    env_path : str or Path, optional
        Path to .env file to load environment variables from.
        If None, tries default ./.env, then falls back to K8s/OS env vars.

    Returns:
    --------
    str: Path to the merged dataset file
    """
    # Load environment with fallback logic
    if env_path is None:
        load_dotenv()
        logger.info("Loading environment from default .env file")
    else:
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")

    # Get environment variables (now guaranteed to exist after validation)
    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID
    dynamic_world_dir = os.environ['dynamic_world_dir']
    vector_lake_file = os.environ['vector_lake_file']
    split_new_dynamic_world_data_dir = os.environ['split_new_dynamic_world_data_dir']
    split_vector_dataset_dir = os.environ['split_vector_dataset_dir']

    all_split_vector_dataset_files = glob.glob(os.path.join(split_vector_dataset_dir, "*.parquet"))

    # Handle optional dynamic_world_data_file
    dynamic_world_data_file = os.environ.get('dynamic_world_data_file')

    # Find most recent existing file
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
    if not all_dynamic_world_files:
        raise FileNotFoundError(f"No .nc files found in {dynamic_world_dir}")

    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)

    logger.debug(f"Dynamic world data file (optional): {dynamic_world_data_file}")
    logger.debug(f"Vector lake file: {vector_lake_file}")
    logger.debug(f"New dynamic world split data dir: {split_new_dynamic_world_data_dir}")
    logger.debug(f"Most recent existing file: {most_recent_dynamic_world_file}")
    logger.debug(f"Found {len(all_split_vector_dataset_files)} split vector files")

    # Check for missing dates
    missing_dates = check_missing_data_in_netcdf(most_recent_dynamic_world_file)
    most_recent_date = max(missing_dates)
    most_recent_date_string = get_date_label(most_recent_date)

    if not missing_dates or len(missing_dates) == 0:
        logger.debug(f"No missing dates found, returning most recent file: {most_recent_dynamic_world_file}")
        return most_recent_dynamic_world_file

    logger.debug(f"Missing dates found: {missing_dates}")
    missing_years = []
    missing_months = []

    for date in missing_dates:
        if date.year not in missing_years:
            missing_years.append(date.year)
        if date.month not in missing_months:
            missing_months.append(date.month)
    logger.debug(f"Missing years: {missing_years}")
    logger.debug(f"Missing months: {missing_months}")

    logger.debug(f"Downloading new dynamic world data from {len(all_split_vector_dataset_files)} split files")

    # Create download filename base
    current_date = str(datetime.now())
    current_date_stamp = current_date.split(' ')[0]
    current_date_stamp = current_date_stamp.replace('-', '_')

    # Ensure download directory exists
    os.makedirs(split_new_dynamic_world_data_dir, exist_ok=True)
    logger.debug(f"New dynamic world data will go to {split_new_dynamic_world_data_dir}")

    # split directory for this download run
    current_split_dynamic_world_dir = os.path.join(split_new_dynamic_world_data_dir, most_recent_date_string)
    os.makedirs(current_split_dynamic_world_dir, exist_ok=True)
    os.makedirs(current_split_dynamic_world_dir, exist_ok=True)
    logger.debug(f"New dynamic world data will go to {current_split_dynamic_world_dir}")


    all_download_filepaths = []
    failed_split_vector_files = []

    # DOWNLOAD INDIVIDUAL NETCDF FILES FROM THE SPLIT FILES
    for i in range(0, len(all_split_vector_dataset_files)):
        split_vector_dataset_file = all_split_vector_dataset_files[i]
        for date in missing_dates:
            current_date_string = get_date_label(date)
            # Extract file number from the split file name
            split_vector_file_name = os.path.basename(split_vector_dataset_file).replace('.parquet', '')
            split_vector_file_number = split_vector_file_name.split('_')[-1]
            logger.debug(f"Downloading new dynamic world data from {date} for split vector file {split_vector_file_name}")
            download_filename = f'dynamic_world_download_{split_vector_file_number}_{current_date_string}.nc'
            download_filepath = os.path.join(current_split_dynamic_world_dir, download_filename)
            logger.debug(f"Downloading new dynamic world data from {date} to {download_filepath}")
            logger.debug(f"Begin download")
            current_year = date.year
            current_month = date.month
            if not os.path.exists(download_filepath):
                try:
                    dl = EarthEngineDownloader(ee_auth=True, logger=logger)
                    ds = dl.download_dw_monthly(
                        vector_dataset=split_vector_dataset_file,
                        name_attribute="id_geohash",
                        years=[current_year],
                        months=[current_month],
                        save_to_file=download_filepath,
                        max_total_requests=500,
                        n_parallel=1,
                    )
                    logger.debug(f"Finished downloading to {download_filepath}")
                    all_download_filepaths.append(download_filepath)
                except Exception as e:
                    logger.error(f"Failed to download data for {split_vector_dataset_file}: {e}")
                    failed_split_vector_files.append(split_vector_dataset_file)
            else:
                logger.debug(f"Already have {download_filepath}")



    logger.info(f"Successfully downloaded {len(all_download_filepaths)} files")
    if failed_split_vector_files:
        logger.warning(f"{len(failed_split_vector_files)} files failed to download: {failed_split_vector_files}")

    if not all_download_filepaths:
        raise RuntimeError("No files were successfully downloaded")

    # Merge the existing and all new data
    logger.debug("Merging existing data with all new downloaded files...")

    # Open existing dataset once
    existing_ds = xr.open_dataset(most_recent_dynamic_world_file)
    existing_lake_ids = set(existing_ds.id_geohash.values)
    existing_dates = set(pd.to_datetime(existing_ds.date.values))

    # List to collect all new datasets
    all_new_datasets = []
    total_new_dates = set()
    total_new_lakes = set()

    # TODO not the download filepaths, all the files under the directory
    # current_split_dynamic_world_dir
    all_download_filepaths = glob.glob(os.path.join(current_split_dynamic_world_dir, "*.nc"))

    # Process each downloaded file
    for i, download_filepath in enumerate(all_download_filepaths, 1):
        logger.info(f"Processing download file {i}/{len(all_download_filepaths)}: {download_filepath}")

        try:
            # Open the new dataset
            new_ds = xr.open_dataset(download_filepath)

            # Check for date overlap and remove if necessary
            new_dates = set(pd.to_datetime(new_ds.date.values))
            overlapping_dates = existing_dates & new_dates

            if overlapping_dates:
                logger.warning(f"Found {len(overlapping_dates)} overlapping dates in {download_filepath}")
                logger.debug(f"Overlapping dates: {sorted(overlapping_dates)}")
                # Remove overlapping dates
                mask = ~new_ds.date.isin(list(overlapping_dates))
                new_ds = new_ds.sel(date=mask)
                new_dates = set(pd.to_datetime(new_ds.date.values))
                logger.info(f"After removal, {len(new_dates)} new dates remain")

            if len(new_ds.date) == 0:
                logger.warning(f"No new dates in {download_filepath}, skipping")
                new_ds.close()
                continue

            # Track statistics
            total_new_dates.update(new_dates)
            new_lakes = set(new_ds.id_geohash.values)
            total_new_lakes.update(new_lakes)

            # Add to collection
            all_new_datasets.append(new_ds)

            logger.debug(f"  - New dates in this file: {len(new_dates)}")
            logger.debug(f"  - New lakes in this file: {len(new_lakes)}")

        except Exception as e:
            logger.error(f"Failed to process {download_filepath}: {e}")
            continue

    if not all_new_datasets:
        logger.warning("No new data to merge after processing all files")
        existing_ds.close()
        return most_recent_dynamic_world_file

    logger.info(f"Processing complete: {len(all_new_datasets)} datasets to merge")
    logger.info(f"Total unique new dates across all files: {len(total_new_dates)}")
    logger.info(f"Total unique new lakes across all files: {len(total_new_lakes)}")

    # Combine all new datasets along the date dimension
    logger.debug("Concatenating all new datasets...")
    if len(all_new_datasets) == 1:
        combined_new_ds = all_new_datasets[0]
    else:
        combined_new_ds = xr.concat(all_new_datasets, dim="date")

    # Sort by date
    combined_new_ds = combined_new_ds.sortby("date")

    # Remove any duplicate dates that might have appeared across multiple split files
    _, unique_indices = np.unique(combined_new_ds.date.values, return_index=True)
    unique_indices.sort()  # Keep chronological order
    if len(unique_indices) < len(combined_new_ds.date):
        logger.warning(f"Removing {len(combined_new_ds.date) - len(unique_indices)} duplicate dates across split files")
        combined_new_ds = combined_new_ds.isel(date=unique_indices)

    # Merge existing with combined new dataset
    logger.debug("Merging existing dataset with combined new dataset...")
    merged_ds = xr.concat([existing_ds, combined_new_ds], dim="date")

    # Sort by date
    merged_ds = merged_ds.sortby("date")

    # Verify the merge was successful
    final_dates = pd.to_datetime(merged_ds.date.values)
    logger.info(f"Original dates: {len(existing_dates)}")
    logger.info(f"New dates added (unique): {len(total_new_dates)}")
    logger.info(f"Total dates after merge: {len(final_dates)}")
    logger.info(f"Date range after merge: {final_dates.min()} to {final_dates.max()}")

    # Create the final filename with timestamp
    new_dynamic_world_filename = f'lakes_dw_V2d_{most_recent_date_string}.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    # Save the merged dataset
    merged_ds.to_netcdf(new_dynamic_world_data_file)

    logger.info(f"Successfully merged datasets!")
    logger.info(f"Original file: {most_recent_dynamic_world_file}")
    logger.info(f"Number of new files merged: {len(all_download_filepaths)}")
    logger.info(f"Merged file saved as: {new_dynamic_world_data_file}")

    # Print summary statistics about lake coverage
    existing_lakes = set(existing_ds.id_geohash.values)
    merged_lakes = set(merged_ds.id_geohash.values)
    new_lakes_final = merged_lakes - existing_lakes

    logger.info(f"Lake coverage summary:")
    logger.info(f"  - Lakes in existing dataset: {len(existing_lakes)}")
    logger.info(f"  - New lakes added in merge: {len(new_lakes_final)}")
    logger.info(f"  - Total lakes after merge: {len(merged_lakes)}")

    # Close datasets to free memory
    existing_ds.close()
    combined_new_ds.close()
    for ds in all_new_datasets:
        ds.close()
    merged_ds.close()

    return new_dynamic_world_data_file

# downloads split file using the split vector files
def download_new_dynamic_world_data_split_files(env_path=None):
    if env_path is None:
        load_dotenv()
        logger.info("Loading environment from default .env file")
    else:
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")

    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID
    dynamic_world_dir = os.environ['dynamic_world_dir']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
    dynamic_world_data_file = os.environ['dynamic_world_data_file']
    new_dynamic_world_data_dir = os.environ['new_dynamic_world_data_dir']

    logger.debug(f"Dynamic world data file: {dynamic_world_data_file}")
    logger.debug(f"New dynamic world data dir: {new_dynamic_world_data_dir}")

    # Check for missing dates
    missing_dates = check_missing_data_in_netcdf(most_recent_dynamic_world_file)

    if not missing_dates or len(missing_dates) == 0:
        logger.debug(f"No missing dates found in {dynamic_world_data_file}")
        return most_recent_dynamic_world_file

    # Get latest missing date for naming
    latest_missing_date = max(missing_dates)
    latest_missing_date_str = str(latest_missing_date).split(" ")[0].replace('-', '_')
    logger.info(f"Latest missing date: {latest_missing_date_str}")

    # Extract unique years and months from missing dates
    missing_years = []
    missing_months = []
    for date in missing_dates:
        if date.year not in missing_years:
            missing_years.append(date.year)
        if date.month not in missing_months:
            missing_months.append(date.month)

    logger.debug(f"Missing years: {missing_years}")
    logger.debug(f"Missing months: {missing_months}")
    logger.debug(f"Downloading new dynamic world data using split files")

    # Get split vector files directory
    split_vector_dataset_dir = os.environ['split_vector_dataset_dir']
    all_split_vector_files = sorted(glob.glob(os.path.join(split_vector_dataset_dir, "*.parquet")))

    if not all_split_vector_files:
        logger.error(f"No split vector files found in {split_vector_dataset_dir}")
        return None

    logger.info(f"Found {len(all_split_vector_files)} split vector files")

    # Create session directory for this download
    current_split_download_directory = os.path.join(new_dynamic_world_data_dir, f"download_{latest_missing_date_str}")
    os.makedirs(current_split_download_directory, exist_ok=True)
    logger.info(f"Downloading chunks to: {current_split_download_directory}")

    # Download each split file
    downloaded_files = []
    successful_chunks = 0

    for i, split_vector_file in enumerate(all_split_vector_files):
        try:
            # Extract label from filename (assuming format like "vector_chunk_00197.parquet")
            split_filename = os.path.basename(split_vector_file)
            # Get the number/label part (between underscores and before .parquet)
            if 'chunk_' in split_filename:
                chunk_label = split_filename.split('chunk_')[1].replace('.parquet', '')
            else:
                # Fallback: use index with padding
                chunk_label = f"{i + 1:05d}"

            # Create numbered download filename
            download_filename = f'dynamic_world_download_{chunk_label}_{latest_missing_date_str}.nc'
            download_filepath = os.path.join(current_split_download_directory, download_filename)

            # Skip if already downloaded
            if os.path.exists(download_filepath):
                logger.info(f"Chunk {chunk_label} already exists, skipping: {download_filepath}")
                downloaded_files.append(download_filepath)
                successful_chunks += 1
                continue

            logger.info(f"Downloading chunk {chunk_label} ({i + 1}/{len(all_split_vector_files)})")
            logger.debug(f"  Vector file: {split_vector_file}")
            logger.debug(f"  Output: {download_filepath}")

            # Initialize downloader
            dl = EarthEngineDownloader(ee_auth=True, logger=logger)

            # Download using the split vector file
            ds = dl.download_dw_monthly(
                vector_dataset=split_vector_file,  # Use the split file directly
                name_attribute="id_geohash",
                years=missing_years,
                months=missing_months,
                save_to_file=download_filepath,
                max_total_requests=500,
                n_parallel=1,
            )

            if ds is not None:
                logger.info(f"✓ Successfully downloaded chunk {chunk_label}")
                downloaded_files.append(download_filepath)
                successful_chunks += 1
            else:
                logger.warning(f"✗ Chunk {chunk_label} returned no data")

        except Exception as e:
            logger.error(f"Failed to download chunk {i + 1}: {e}")
            continue

    if not downloaded_files:
        logger.error("No chunks were successfully downloaded")
        return None

    logger.info(f"Successfully downloaded {successful_chunks}/{len(all_split_vector_files)} chunks")

    # Merge all downloaded chunks
    logger.info("Merging downloaded chunks...")


    # TODO combine here
    new_dynamic_world_data = combine_new_dynamic_world_data_with_latest(path_to_new_data=current_split_download_directory, env_path=env_path)
    logger.info(f"New data is {new_dynamic_world_data}")





