#!/usr/bin/env python3
import zarr
import numpy as np
from datetime import datetime, timedelta

zarr_path = "/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/dw_downloads/dynamic_world_data_2026-05-06_17:23:29.832165.zarr"


def print_zarr_dates(zarr_path):
    """Print all date information from a Zarr file"""
    root = zarr.open(zarr_path, mode='r')

    print(f"\n📁 File: {zarr_path}\n")
    print("Available arrays:", list(root.keys()))
    print("Attributes:", dict(root.attrs))

    # Look for date/time coordinates
    for coord_name in ['date', 'time', 'datetime', 'timestamp']:
        if coord_name in root:
            data = root[coord_name][:]
            print(f"\n📅 Found '{coord_name}':")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Data sample: {data[:5]}")

            # Try to convert to readable dates
            if np.issubdtype(data.dtype, np.number):
                try:
                    # Try days since epoch
                    dates = [datetime(1970, 1, 1) + timedelta(days=int(d)) for d in data[:10]]
                    print(f"  As dates (days since 1970): {[d.strftime('%Y-%m-%d') for d in dates]}")
                except:
                    pass

                try:
                    # Try seconds since epoch
                    dates = [datetime(1970, 1, 1) + timedelta(seconds=float(d)) for d in data[:10]]
                    print(f"  As dates (seconds since 1970): {[d.strftime('%Y-%m-%d') for d in dates]}")
                except:
                    pass

            # Show values
            print(f"\n  📆 All dates from '{coord_name}':")
            for i, val in enumerate(data):
                print(f"    {i}: {val}")

    # Look for date in attributes
    for key, value in root.attrs.items():
        if 'date' in key.lower() or 'time' in key.lower():
            print(f"\n📝 Attribute '{key}': {value}")


if __name__ == "__main__":
    print_zarr_dates(zarr_path)