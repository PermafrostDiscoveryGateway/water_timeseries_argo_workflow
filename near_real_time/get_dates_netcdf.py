import netCDF4 as nc
import pandas as pd
from netCDF4 import num2date
from datetime import datetime
from dateutil.relativedelta import relativedelta


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


# Choose one of the methods above - here's the recommended approach:

dynamic_world_data = '/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/data/input/lakes_dw_V2d.nc'

# Method 1: Using xarray (simplest and most reliable)
print("=== Using xarray to extract dates ===")
import xarray as xr

ds = xr.open_dataset(dynamic_world_data)
dates = pd.to_datetime(ds.date.values)
ds.close()

print(f"Number of dates: {len(dates)}")
print(f"First 5 dates: {dates[:5]}")
print(f"Last 5 dates: {dates[-5:]}")
print(f"\nAll dates:")
for date in dates:
    print(f"  {date.strftime('%Y-%m-%d')}")

# Get as list of tuples (year, month)
year_month_list = [(d.year, d.month) for d in dates]
print(f"\nYear-month pairs: {year_month_list}")

# Get as dictionary for quick lookup
date_dict = {f"{d.year}-{d.month:02d}": d for d in dates}
print(f"\nAvailable periods: {list(date_dict.keys())}")

# Check if all are on 1st of month
all_first = all(d.day == 1 for d in dates)
print(f"\nAll dates on 1st of month? {all_first}")


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


# Usage
dynamic_world_data = '/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/data/input/lakes_dw_V2d.nc'
missing = check_missing_data_in_netcdf(dynamic_world_data)
print(missing)
print('found missing dates')

print("download these dates for all lake_ids")