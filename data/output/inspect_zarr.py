#!/usr/bin/env python
"""
Inspect Zarr file contents and structure.
Usage: python inspect_zarr.py <path_to_zarr_file>
Example: python inspect_zarr.py /path/to/breakpoints_2026-06.zarr
"""

import sys
import os
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path


def inspect_zarr(zarr_path: str, show_data: bool = True, max_rows: int = 10):
    """
    Inspect a Zarr file and print its contents.

    Args:
        zarr_path: Path to the Zarr directory
        show_data: If True, show sample data rows
        max_rows: Maximum number of rows to display
    """
    print("=" * 80)
    print(f"ZARR FILE INSPECTION")
    print("=" * 80)
    print(f"Path: {zarr_path}")

    # Check if path exists
    if not os.path.exists(zarr_path):
        print(f"❌ ERROR: Path does not exist: {zarr_path}")
        return

    # Check if it's a directory
    if not os.path.isdir(zarr_path):
        print(f"❌ ERROR: Path is not a directory: {zarr_path}")
        return

    try:
        # Open the Zarr file
        ds = xr.open_zarr(zarr_path)

        # Basic info
        print(f"\n📊 BASIC INFO:")
        print(f"  Dimensions: {dict(ds.dims)}")
        print(f"  Data variables: {list(ds.data_vars)}")
        print(f"  Coordinates: {list(ds.coords.keys())}")

        # Check for id_geohash dimension
        if 'id_geohash' in ds.dims:
            n_ids = len(ds.id_geohash)
            print(f"  Number of IDs: {n_ids:,}")
            print(f"  Sample IDs: {ds.id_geohash.values[:5].tolist()}")
        else:
            print(f"  ⚠️ No 'id_geohash' dimension found")

        # Check for date dimension
        if 'date' in ds.coords:
            dates = pd.to_datetime(ds.date.values)
            print(f"  Date range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
            print(f"  Number of dates: {len(dates)}")

        # Check drainage_confidence
        print(f"\n📈 CONFIDENCE LEVELS:")
        if 'drainage_confidence' in ds.data_vars:
            conf = ds.drainage_confidence.values
            # Flatten if needed
            conf_flat = conf.flatten() if conf.ndim > 1 else conf

            low = int(np.sum(conf_flat == 1))
            medium = int(np.sum(conf_flat == 2))
            high = int(np.sum(conf_flat == 3))
            nan_count = int(np.sum(np.isnan(conf_flat)))
            zero_count = int(np.sum(conf_flat == 0))
            total = len(conf_flat)

            print(f"  Low (1): {low:,}")
            print(f"  Medium (2): {medium:,}")
            print(f"  High (3): {high:,}")
            print(f"  Zero (0): {zero_count:,}")
            print(f"  NaN: {nan_count:,}")
            print(f"  Total: {total:,}")

            # Valid data (non-zero, non-NaN)
            valid = low + medium + high
            print(f"  Valid breakpoints: {valid:,} ({valid / total * 100:.1f}%)")
        else:
            print(f"  ⚠️ No 'drainage_confidence' variable found")

        # Check other variables
        print(f"\n📦 VARIABLE STATS:")
        for var_name in ds.data_vars:
            if var_name == 'drainage_confidence':
                continue
            try:
                data = ds[var_name].values
                data_flat = data.flatten() if data.ndim > 1 else data
                if np.issubdtype(data.dtype, np.number):
                    valid_data = data_flat[~np.isnan(data_flat)]
                    if len(valid_data) > 0:
                        print(f"  {var_name}:")
                        print(f"    min: {np.min(valid_data):.4f}")
                        print(f"    max: {np.max(valid_data):.4f}")
                        print(f"    mean: {np.mean(valid_data):.4f}")
                        print(f"    std: {np.std(valid_data):.4f}")
                        print(f"    non-NaN count: {len(valid_data):,}")
                    else:
                        print(f"  {var_name}: All NaN")
                else:
                    # Non-numeric data
                    unique_vals = np.unique(data_flat)
                    print(f"  {var_name}: {len(unique_vals)} unique values")
                    if len(unique_vals) <= 10:
                        print(f"    Values: {unique_vals.tolist()}")
            except Exception as e:
                print(f"  {var_name}: Error reading - {e}")

        # Show sample data
        if show_data:
            print(f"\n📋 SAMPLE DATA (first {max_rows} rows):")
            try:
                # Convert to DataFrame
                df = ds.to_dataframe()
                print(df.head(max_rows))
                print(f"\nShape: {df.shape}")
            except Exception as e:
                print(f"Could not convert to DataFrame: {e}")
                # Try alternative method
                try:
                    for i, idx in enumerate(list(ds.id_geohash.values)[:max_rows]):
                        print(f"\nID: {idx}")
                        for var in ds.data_vars:
                            val = ds[var].sel(id_geohash=idx).values
                            if isinstance(val, np.ndarray):
                                val = val.flatten()
                            print(f"  {var}: {val}")
                except Exception as e2:
                    print(f"Could not display sample data: {e2}")

        # File size
        print(f"\n💾 FILE INFO:")
        try:
            total_size = 0
            for f in Path(zarr_path).rglob('*'):
                if f.is_file():
                    total_size += f.stat().st_size
            size_gb = total_size / (1024 ** 3)
            size_mb = total_size / (1024 ** 2)
            if size_gb > 1:
                print(f"  Total size: {size_gb:.2f} GB")
            elif size_mb > 1:
                print(f"  Total size: {size_mb:.2f} MB")
            else:
                print(f"  Total size: {total_size / 1024:.2f} KB")

            # Count files
            n_files = len(list(Path(zarr_path).rglob('*')))
            print(f"  Number of files: {n_files}")
        except Exception as e:
            print(f"  Could not calculate size: {e}")

        # Close the dataset
        ds.close()

    except Exception as e:
        print(f"❌ ERROR reading Zarr file: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""

    # TODO have it intelligently find new inputs and print out analysis on all of them

    zarr_path = '/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/data/output/EURASIA3/breakpoint_zarr/breakpoints_2026-06.zarr'
    show_data = True

    if len(sys.argv) > 2 and sys.argv[2] == '--no-data':
        show_data = False

    inspect_zarr(zarr_path, show_data=show_data)


if __name__ == "__main__":
    main()