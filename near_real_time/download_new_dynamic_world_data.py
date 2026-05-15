import netCDF4 as nc
import pandas as pd
from netCDF4 import num2date
from datetime import datetime
from loguru import logger
import os
import glob
import sys
import xarray as xr
from dotenv import load_dotenv
from water_timeseries.downloader import EarthEngineDownloader
import water_timeseries.dataset


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


def load_environment(env_path=None):
    """
    Load environment variables with fallback priority:
    1. Provided env_path argument (.env file path)
    2. Default ./.env file
    3. Kubernetes/OS environment variables (already present)

    Returns:
        bool: True if environment variables were loaded from a .env file, False if using K8s/OS
    """
    loaded_from_file = False

    # Priority 1: Provided env_path argument
    if env_path:
        env_path_obj = Path(env_path)
        if env_path_obj.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            logger.info(f"Loaded environment from provided .env: {env_path}")
            loaded_from_file = True
        else:
            logger.warning(f".env file not found at {env_path}, checking other sources")

    # Priority 2: Default .env file in current directory
    if not loaded_from_file:
        default_env = Path.cwd() / ".env"
        if default_env.exists():
            load_dotenv(dotenv_path=default_env, override=False)
            logger.info(f"Loaded environment from default .env: {default_env}")
            loaded_from_file = True
        else:
            logger.info("No .env file found, using Kubernetes/OS environment variables")

    # Validate required environment variables
    required_vars = [
        'project',
        'dynamic_world_dir',
        'vector_lake_file',
        'new_dynamic_world_data_dir'
    ]

    missing_vars = []
    for var in required_vars:
        if var not in os.environ:
            missing_vars.append(var)

    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise EnvironmentError(error_msg)

    # Log which source is providing each variable (debug)
    logger.debug("Environment configuration:")
    for var in required_vars + ['dynamic_world_data_file']:
        if var in os.environ:
            source = ".env" if loaded_from_file else "K8s/OS"
            # Don't log full values if they contain sensitive data
            if any(sensitive in var.lower() for sensitive in ['key', 'secret', 'password', 'token']):
                logger.debug(f"  {var} = *** (source: {source})")
            else:
                logger.debug(f"  {var} = {os.environ[var]} (source: {source})")

    return loaded_from_file


def download_new_dynamic_world_data(env_path=None):
    """
    Download new Dynamic World data for missing dates.

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
    load_environment(env_path)

    # Get environment variables (now guaranteed to exist after validation)
    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID
    dynamic_world_dir = os.environ['dynamic_world_dir']
    vector_lake_file = os.environ['vector_lake_file']
    new_dynamic_world_data_dir = os.environ['new_dynamic_world_data_dir']

    # Handle optional dynamic_world_data_file
    dynamic_world_data_file = os.environ.get('dynamic_world_data_file')

    # Find most recent existing file
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
    if not all_dynamic_world_files:
        raise FileNotFoundError(f"No .nc files found in {dynamic_world_dir}")

    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)

    logger.debug(f"Dynamic world data file (optional): {dynamic_world_data_file}")
    logger.debug(f"Vector lake file: {vector_lake_file}")
    logger.debug(f"New dynamic world data dir: {new_dynamic_world_data_dir}")
    logger.debug(f"Most recent existing file: {most_recent_dynamic_world_file}")

    # Check for missing dates
    missing_dates = check_missing_data_in_netcdf(most_recent_dynamic_world_file)

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

    logger.debug(f"Downloading new dynamic world data")

    # Create download filename
    current_date = str(datetime.now())
    current_date_stamp = current_date.split(' ')[0]
    current_date_stamp = current_date_stamp.replace('-', '_')
    download_filename = 'dynamic_world_download_' + current_date_stamp + '.nc'
    download_filepath = os.path.join(new_dynamic_world_data_dir, download_filename)

    # Ensure download directory exists
    os.makedirs(new_dynamic_world_data_dir, exist_ok=True)

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


# For testing purposes
# if __name__ == "__main__":
#     # Allow command line argument for .env path when run directly
#     env_path = sys.argv[1] if len(sys.argv) > 1 else None
#     result = download_new_dynamic_world_data(env_path=env_path)
#     print(f"Download complete: {result}")