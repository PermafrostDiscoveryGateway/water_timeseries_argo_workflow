import xarray as xr
import zarr
from pathlib import Path
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, List, Any


def get_zarr_dates(
        zarr_path: Union[str, Path],
        time_coord_name: Optional[str] = None,
        as_pandas: bool = True
) -> np.ndarray:
    """
    Extract all dates/times from a Zarr dataset.

    Parameters:
    -----------
    zarr_path : str or Path
        Path to the .zarr directory
    time_coord_name : str, optional
        Name of the time coordinate if known (e.g., 'time', 'Time', 'valid_time')
        If None, automatically tries to detect it
    as_pandas : bool, default=True
        If True, returns pandas DatetimeIndex or array of Timestamps
        If False, returns raw values from the dataset

    Returns:
    --------
    np.ndarray : Array of dates/times (pandas Timestamps if as_pandas=True)

    Raises:
    -------
    ValueError : If no time dimension is found in the dataset
    """
    zarr_path = Path(zarr_path)

    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path not found: {zarr_path}")

    # Open the Zarr dataset
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
    except:
        # Try without consolidated metadata
        ds = xr.open_zarr(zarr_path, consolidated=False)

    # Find the time coordinate
    if time_coord_name is None:
        # Common names for time dimensions
        time_candidates = ['time', 'Time', 't', 'datetime', 'date', 'valid_time',
                           'forecast_time', 'reference_time', 'init_time']
        time_coord_name = None

        for candidate in time_candidates:
            if candidate in ds.coords:
                time_coord_name = candidate
                break

        # If still not found, look for any dimension with time-like units
        if time_coord_name is None:
            for coord_name, coord_var in ds.coords.items():
                if hasattr(coord_var, 'encoding') and 'units' in coord_var.encoding:
                    units = coord_var.encoding['units']
                    if 'since' in units.lower():
                        time_coord_name = coord_name
                        break

    if time_coord_name is None:
        raise ValueError(
            f"No time dimension found in {zarr_path}. "
            f"Available coordinates: {list(ds.coords.keys())}"
        )

    # Extract the time values
    time_var = ds[time_coord_name]
    time_values = time_var.values

    if as_pandas:
        try:
            # Try to convert to pandas datetime
            dates = pd.to_datetime(time_values)
        except:
            # Handle cftime objects if present
            try:
                import cftime
                if isinstance(time_values[0], cftime.datetime):
                    # Convert cftime to pandas
                    dates = pd.DatetimeIndex([pd.Timestamp(t) for t in time_values])
                else:
                    # Try using time units if available in encoding
                    if hasattr(time_var, 'encoding') and 'units' in time_var.encoding:
                        units = time_var.encoding['units']
                        calendar = time_var.encoding.get('calendar', 'standard')
                        times_num = time_values
                        dates = cftime.num2date(times_num, units, calendar=calendar)
                        # Convert to pandas
                        dates = pd.DatetimeIndex([pd.Timestamp(t) for t in dates])
                    else:
                        dates = time_values
            except:
                dates = time_values
    else:
        dates = time_values

    ds.close()
    return dates


# Alternative version using xarray's built-in capabilities (often simpler)
def get_zarr_dates_xarray(
        zarr_path: Union[str, Path],
        time_coord_name: Optional[str] = None
) -> xr.DataArray:
    """
    Extract dates using xarray's native handling.
    Returns an xarray DataArray with datetime64 dtype if possible.
    """
    zarr_path = Path(zarr_path)

    # Open the dataset
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
    except:
        ds = xr.open_zarr(zarr_path, consolidated=False)

    # Find time coordinate
    if time_coord_name is None:
        # Auto-detect
        for candidate in ['time', 'Time', 't', 'datetime', 'date', 'valid_time']:
            if candidate in ds.coords:
                time_coord_name = candidate
                break

    if time_coord_name is None:
        raise ValueError(f"No time coordinate found. Available: {list(ds.coords.keys())}")

    # Xarray automatically converts to datetime64 if possible
    dates = ds[time_coord_name]

    # Don't close yet if you need the dataset; alternatively, copy the values
    dates_array = dates.copy()
    ds.close()

    return dates_array


# Lightweight version using zarr directly (no xarray, faster for large datasets)
def get_zarr_dates_zarr_only(
        zarr_path: Union[str, Path],
        time_coord_name: str = 'time'
) -> np.ndarray:
    """
    Extract dates using only zarr (faster, no xarray overhead).
    Use if you know the exact time coordinate name.
    """
    import zarr

    zarr_path = Path(zarr_path)
    root = zarr.open_group(zarr_path, mode='r')

    if time_coord_name not in root:
        # Try to find any time-like coordinate
        for key in root.keys():
            if isinstance(root[key], zarr.Array) and any(t in key.lower() for t in ['time', 'date']):
                time_coord_name = key
                break
        else:
            raise ValueError(f"No time coordinate found. Available: {list(root.keys())}")

    # Read the time values
    time_values = root[time_coord_name][:]

    # Try to parse as dates if they're numeric with units
    time_attrs = root[time_coord_name].attrs.asdict()
    if 'units' in time_attrs and 'since' in time_attrs['units']:
        import cftime
        units = time_attrs['units']
        calendar = time_attrs.get('calendar', 'standard')
        dates = cftime.num2date(time_values, units, calendar=calendar)
    else:
        dates = time_values

    return np.array(dates)


# Example usage and utility functions
def get_unique_dates(zarr_path: Union[str, Path]) -> np.ndarray:
    """Get unique dates from a Zarr dataset (handles duplicate times)."""
    dates = get_zarr_dates(zarr_path)
    if isinstance(dates, pd.DatetimeIndex):
        return dates.unique()
    else:
        return np.unique(dates)


def get_date_range(zarr_path: Union[str, Path]) -> tuple:
    """Get the min and max dates from a Zarr dataset."""
    dates = get_zarr_dates(zarr_path)
    return dates.min(), dates.max()


def get_dates_summary(zarr_path: Union[str, Path]) -> dict:
    """Get a summary dictionary of date information."""
    dates = get_zarr_dates(zarr_path)

    summary = {
        'count': len(dates),
        'min': dates.min(),
        'max': dates.max(),
    }

    # Add unique count if pandas datetime
    if isinstance(dates, pd.DatetimeIndex):
        summary['unique_count'] = dates.nunique()
        summary['has_duplicates'] = dates.nunique() < len(dates)

    return summary

def inspect_zarr_dates(zarr_path):
    """
    Inspect a Zarr dataset and report available dates/times.

    Parameters:
    zarr_path: str or Path - path to the .zarr directory
    """
    zarr_path = Path(zarr_path)

    if not zarr_path.exists():
        print(f"Error: Path {zarr_path} does not exist")
        return

    try:
        # Open the Zarr dataset with xarray (handles geospatial data well)
        ds = xr.open_zarr(zarr_path, consolidated=True)

        print(f"\n{'=' * 60}")
        print(f"Dataset: {zarr_path.name}")
        print(f"{'=' * 60}\n")

        # Look for time dimension (common names: time, Time, t, datetime)
        time_coords = ['time', 'Time', 't', 'datetime', 'date', 'valid_time']
        found_time = None

        for coord in time_coords:
            if coord in ds.coords:
                found_time = coord
                break

        if found_time:
            time_var = ds[found_time]
            print(f"✓ Time dimension found: '{found_time}'")
            print(f"  Shape: {time_var.shape}")
            print(f"  Data type: {time_var.dtype}")

            # Convert to pandas datetime if possible
            try:
                if hasattr(time_var, 'values'):
                    times = time_var.values
                    # Handle different time encodings
                    if hasattr(time_var, 'encoding') and 'units' in time_var.encoding:
                        import cftime
                        times = cftime.num2date(times, time_var.encoding['units'])

                    print(f"\n📅 Date range in dataset:")
                    print(f"  Start: {times[0]}")
                    print(f"  End: {times[-1]}")
                    print(f"  Number of timesteps: {len(times)}")

                    # Show first and last few dates
                    print(f"\n  First 5 dates:")
                    for t in times[:5]:
                        print(f"    - {t}")

                    if len(times) > 10:
                        print(f"  ...")
                        print(f"  Last 5 dates:")
                        for t in times[-5:]:
                            print(f"    - {t}")

                    # Optional: Check for gaps in dates
                    if len(times) > 1:
                        try:
                            import pandas as pd
                            pd_times = pd.to_datetime(times)
                            gaps = pd_times[1:] - pd_times[:-1]
                            if gaps.max() > gaps.median() * 2:
                                print(f"\n  ⚠️ Irregular timesteps detected (max gap: {gaps.max()})")
                        except:
                            pass

            except Exception as e:
                print(f"  Could not parse dates: {e}")
                print(f"  Raw time values: {time_var.values[:5]}...")
        else:
            print("❌ No explicit time dimension found in coordinates")
            print("\nAvailable coordinates:")
            for coord in ds.coords:
                print(f"  - {coord}")

        # Print additional geospatial info
        print(f"\n🌍 Geospatial info:")
        if 'latitude' in ds.coords or 'lat' in ds.coords:
            lat = ds.coords.get('latitude', ds.coords.get('lat'))
            print(f"  Latitude: {lat.values.min():.2f}° to {lat.values.max():.2f}°")
        if 'longitude' in ds.coords or 'lon' in ds.coords:
            lon = ds.coords.get('longitude', ds.coords.get('lon'))
            print(f"  Longitude: {lon.values.min():.2f}° to {lon.values.max():.2f}°")

        # List all data variables
        print(f"\n📊 Data variables:")
        for var in ds.data_vars:
            print(f"  - {var}: {ds[var].dims}")

        ds.close()

    except Exception as e:
        print(f"Error reading Zarr dataset: {e}")
        print("\nTrying with different method (no consolidation)...")
        try:
            ds = xr.open_zarr(zarr_path, consolidated=False)
            print("✓ Successfully opened without consolidated metadata")
            # Repeat the logic above...
        except Exception as e2:
            print(f"Still failed: {e2}")


if __name__ == "__main__":
    # Usage examples:

    # Option 1: Direct path to .zarr folder
    # zarr_dataset_path = "path/to/your/data.zarr"  # CHANGE THIS
    zarr_dataset_path = "/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/utils/2016_6_102025_dw_download.zarr"

    # Option 2: Search for .zarr folders in current directory
    # import glob
    # zarr_folders = glob.glob("*.zarr")
    # for zarr_path in zarr_folders:
    #     inspect_zarr_dates(zarr_path)

    inspect_zarr_dates(zarr_dataset_path)