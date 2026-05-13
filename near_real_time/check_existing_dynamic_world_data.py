import os
import h5py
import numpy as np

import netCDF4 as nc
import xarray as xr

import pandas as pd
import netCDF4 as nc
from netCDF4 import num2date


def get_all_dates_simple(netcdf_path):
    """
    Simple function to get all dates as pandas datetime objects.
    Confirms all dates are on the 1st of the month.
    """
    with nc.Dataset(netcdf_path, 'r') as ds:
        # Convert numeric dates to datetime
        dates = num2date(
            ds.variables['date'][:],
            units=ds.variables['date'].units,
            calendar=getattr(ds.variables['date'], 'calendar', 'standard')
        )

        # Convert to pandas datetime
        dates_pd = pd.to_datetime(dates)

        # Verify they're all on the 1st
        if all(dates_pd.dt.day == 1):
            print("✓ All dates are on the 1st of the month")
        else:
            print("⚠ Warning: Not all dates are on the 1st of the month")

        return dates_pd



dynamic_world_data = '/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/data/input/lakes_dw_V2d.nc'

# Usage
dates = get_all_dates_simple(dynamic_world_data)

# Get as list of tuples (year, month)
year_month_list = [(d.year, d.month) for d in dates]
print(f"\nYear-month pairs: {year_month_list}")

# Get as dictionary for quick lookup
date_dict = {f"{d.year}-{d.month:02d}": d for d in dates}
print(f"\nAvailable periods: {list(date_dict.keys())}")

print("=== Opening file with netCDF4 ===")
dataset = nc.Dataset(dynamic_world_data, 'r')

print("\n=== File Information ===")
print(f"File format: {dataset.file_format}")
print(f"Dimensions: {dataset.dimensions}")
print(f"Number of variables: {len(dataset.variables)}")

print("\n=== Variables (what you called 'columns') and their data types ===")
for var_name in dataset.variables:
    var = dataset.variables[var_name]
    print(f"\nVariable: {var_name}")
    print(f"  Data type: {var.dtype}")
    print(f"  Dimensions: {var.dimensions}")
    print(f"  Shape: {var.shape}")

    # Print attributes if they exist
    if hasattr(var, 'units'):
        print(f"  Units: {var.units}")
    if hasattr(var, 'long_name'):
        print(f"  Long name: {var.long_name}")
    if hasattr(var, '_FillValue'):
        print(f"  Fill value: {var._FillValue}")

dataset.close()

print("\n=== Opening with xarray (alternative method) ===")
# With xarray, we need to be careful with large files
ds = xr.open_dataset(dynamic_world_data, chunks={})  # chunks={} avoids auto-chunking

print("\n=== Dataset Overview ===")
print(ds)

print("\n=== Data Variables Summary ===")
for var_name in ds.data_vars:
    print(f"\n{var_name}:")
    print(f"  Type: {ds[var_name].dtype}")
    print(f"  Shape: {ds[var_name].shape}")
    print(f"  Dimensions: {ds[var_name].dims}")

    # Show first few values for coordinates/small variables
    if ds[var_name].size < 100:  # Only show if small
        print(f"  Sample values: {ds[var_name].values[:5]}")

print("\n=== Coordinates ===")
for coord_name in ds.coords:
    print(f"\n{coord_name}:")
    print(f"  Type: {ds[coord_name].dtype}")
    print(f"  Shape: {ds[coord_name].shape}")
    if ds[coord_name].size < 100:
        print(f"  Values: {ds[coord_name].values}")

# Close the dataset
ds.close()

dynamic_world_data = '/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/data/input/lakes_dw_V2d.nc'

# Step 1: Check if file exists and its size
print("=== File Information ===")
if os.path.exists(dynamic_world_data):
    file_size = os.path.getsize(dynamic_world_data)
    print(f"File exists: Yes")
    print(f"File size: {file_size} bytes ({file_size / (1024 * 1024):.2f} MB)")

    if file_size == 0:
        print("ERROR: File is empty (0 bytes)")
    elif file_size < 1000:  # Less than 1KB is suspicious
        print("WARNING: File is very small, likely corrupt or not a real NetCDF")
else:
    print(f"File does NOT exist at: {dynamic_world_data}")
    exit()

# Step 2: Try to read the first few bytes to see what type of file it is
print("\n=== File Header Check ===")
with open(dynamic_world_data, 'rb') as f:
    header = f.read(100)
    print(f"First 50 bytes (hex): {header[:50].hex()}")
    print(f"First 50 bytes (ascii): {header[:50]}")

    # Check for NetCDF magic numbers
    if header[:3] == b'CDF':
        print("This appears to be a classic NetCDF file")
    elif header[:8] == b'\x89HDF\r\n\x1a\n':
        print("This appears to be an HDF5 file (NetCDF4 uses HDF5)")
    else:
        print("This does NOT start with standard NetCDF/HDF5 signatures")
        print("File might be corrupted or not a NetCDF file")

# Step 3: Try different methods to open
print("\n=== Attempting to open with different methods ===")

# Method A: Try netCDF4 with verbose error
try:
    import netCDF4 as nc

    print("Trying netCDF4...")
    dataset = nc.Dataset(dynamic_world_data, 'r')
    print("SUCCESS with netCDF4!")
    dataset.close()
except Exception as e:
    print(f"netCDF4 failed: {e}")

# Method B: Try h5py directly (since NetCDF4 is built on HDF5)
try:
    import h5py

    print("\nTrying h5py (direct HDF5 read)...")
    with h5py.File(dynamic_world_data, 'r') as f:
        print("SUCCESS! File is valid HDF5/NetCDF4")
        print(f"Keys in file: {list(f.keys())}")
except Exception as e:
    print(f"h5py failed: {e}")

# Method C: Try reading as raw HDF5 to salvage any data
try:
    print("\nTrying to salvage any readable data...")
    with h5py.File(dynamic_world_data, 'r', driver='core', backing_store=False) as f:
        print("File can be opened in core driver mode")
except Exception as e:
    print(f"Even core driver failed: {e}")

# Step 4: Check if file is from Google Cloud Storage (might not be fully downloaded)
print("\n=== Google Cloud Storage Check ===")
print("If this file was downloaded from GCS, it might be:")
print("1. Partially downloaded (corrupted)")
print("2. A symlink or placeholder")
print("3. Not fully transferred")

# Step 5: Try to fix if it's a common issue
print("\n=== Possible Solutions ===")
print("1. Re-download the file from Google Cloud Storage")
print("2. Check if the file is compressed (try adding '.gz' to filename)")
print("3. Verify the file checksum if available")
print("4. Try opening with a different library: `import scipy.io.netcdf`")