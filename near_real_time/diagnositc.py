import xarray as xr
import os
import glob
from dotenv import load_dotenv
import numpy as np
load_dotenv()

dynamic_world_dir = os.environ.get('dynamic_world_dir')

# Find the two files
existing_file = None
new_file = None

for f in glob.glob(os.path.join(dynamic_world_dir, "*.nc")):
    if "lakes_dw_V2d.nc" in f and "2025" not in f:
        existing_file = f
    elif "2025_09_01" in f:
        new_file = f

print("=" * 60)
print("FILE COMPARISON")
print("=" * 60)

import xarray as xr
import os

old_file = os.path.join(dynamic_world_dir, "lakes_dw_V2d.nc")
new_file = os.path.join(dynamic_world_dir, "lakes_dw_V2d_2025_09_01.nc")
# Check the compression settings on both files
old = xr.open_dataset(old_file, decode_times=False)
new = xr.open_dataset(new_file, decode_times=False)

print("Old file encoding:", old.bare.encoding)
print("New file encoding:", new.bare.encoding)

if existing_file:
    print(f"\nExisting file: {os.path.basename(existing_file)}")
    print(f"  Size: {os.path.getsize(existing_file) / (1024 ** 3):.2f} GB")
    with xr.open_dataset(existing_file, decode_times=False) as ds:
        print(f"  Lakes: {len(ds.id_geohash):,}")
        print(f"  Dates: {len(ds.date)}")
        print(f"  Date range: {ds.date.values.min()} to {ds.date.values.max()}")
        print(f"  Variables: {list(ds.data_vars.keys())}")
        # Check data density
        for var in list(ds.data_vars.keys())[:1]:
            data = ds[var].values
            nan_count = np.isnan(data).sum()
            print(f"  {var} NaN %: {nan_count / data.size * 100:.1f}%")

if new_file:
    print(f"\nNew file: {os.path.basename(new_file)}")
    print(f"  Size: {os.path.getsize(new_file) / (1024 ** 3):.2f} GB")
    with xr.open_dataset(new_file, decode_times=False) as ds:
        print(f"  Lakes: {len(ds.id_geohash):,}")
        print(f"  Dates: {len(ds.date)}")
        print(f"  Date range: {ds.date.values.min()} to {ds.date.values.max()}")
        print(f"  Variables: {list(ds.data_vars.keys())}")
        # Check data density
        for var in list(ds.data_vars.keys())[:1]:
            data = ds[var].values
            nan_count = np.isnan(data).sum()
            print(f"  {var} NaN %: {nan_count / data.size * 100:.1f}%")

        # Check if lake IDs match
        existing_ids = set()
        if existing_file:
            with xr.open_dataset(existing_file, decode_times=False) as ds_existing:
                existing_ids = set(ds_existing.id_geohash.values[:1000])

        new_ids_sample = set(ds.id_geohash.values[:1000])
        overlap = existing_ids & new_ids_sample
        print(f"\n  Lake ID overlap (first 1000): {len(overlap)}/1000")

print("\n" + "=" * 60)