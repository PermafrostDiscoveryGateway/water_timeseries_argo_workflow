#!/usr/bin/env python3
import os
import zarr
import xarray as xr
import numpy as np
from pathlib import Path

# Disable async
os.environ["ZARR_ASYNC"] = "0"

zarr_path = "/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/dw_downloads/dynamic_world_data_2026-05-06_17:23:29.832165.zarr"
output_zarr_v2 = "/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/dw_downloads/dynamic_world_data_full_v2.zarr"

print(f"Converting Zarr file...")

try:
    # Open original
    root = zarr.open(zarr_path, mode='r')

    # Collect data arrays and coordinates
    data_vars = {}
    coords = {}

    # Get id_geohash
    if 'id_geohash' in root:
        coords['id_geohash'] = root['id_geohash'][:]
        print(f"Found id_geohash with {len(coords['id_geohash'])} entries")

    # Get date/time coordinate - RENAME to 'date' instead of 'time'
    if 'date' in root:
        date_data = root['date'][:]
        # Try to convert to datetime if it's numeric
        try:
            if np.issubdtype(date_data.dtype, np.number):
                from datetime import datetime, timedelta

                date_array = [datetime(1970, 1, 1) + timedelta(days=int(d)) for d in date_data]
                coords['date'] = date_array  # CHANGED: 'date' instead of 'time'
                print(f"Converted {len(date_array)} numeric dates to datetime")
            else:
                coords['date'] = date_data  # CHANGED: 'date' instead of 'time'
                print(f"Found {len(date_data)} date entries")
        except Exception as e:
            print(f"Warning: Could not convert dates: {e}")
            coords['date'] = date_data  # CHANGED: 'date' instead of 'time'

    # Convert each variable
    for key in root.keys():
        if key not in ['date', 'id_geohash']:
            zarr_array = root[key]
            data = zarr_array[:]

            # Determine dimensions - use 'date' instead of 'time'
            if len(data.shape) == 1:
                dims = ['id_geohash']
            else:
                dims = ['id_geohash', 'date']  # CHANGED: 'date' instead of 'time'

            data_vars[key] = (dims, data)
            print(f"  {key}: {dims}")

    # Create xarray dataset
    ds = xr.Dataset(data_vars, coords=coords)

    # Save as Zarr v2
    print(f"Saving to {output_zarr_v2}...")
    ds.to_zarr(output_zarr_v2, mode='w', consolidated=False)
    print("✓ Conversion complete!")

    # Verify
    print("\nVerifying converted file...")
    ds_verify = xr.open_zarr(output_zarr_v2, consolidated=False)
    print(f"✓ Success! Dimensions: {dict(ds_verify.dims)}")
    print(f"  Coordinates: {list(ds_verify.coords.keys())}")
    print(f"  Variables: {list(ds_verify.data_vars.keys())}")

    if 'date' in ds_verify.dims:
        print(f"  ✓ Has 'date' dimension with {len(ds_verify.date)} points")

    print(f"\n✅ Ready to use! Update your config with:")
    print(f"water_dataset_file: {output_zarr_v2}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()