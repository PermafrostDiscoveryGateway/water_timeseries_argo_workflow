#!/usr/bin/env python3
"""Verify that PyCharm is using the correct environment with the Zarr fix"""

import sys
import os
from pathlib import Path


def main():
    print("=" * 60)
    print("Environment Verification")
    print("=" * 60)

    # Check Python executable
    print(f"\nPython executable: {sys.executable}")

    # Check if it's from the first repo
    if "water-timeseries-v2" in sys.executable:
        print("✅ Using environment from water-timeseries-v2")
    else:
        print(f"⚠️  Using different environment: {sys.executable}")
        print("   Consider switching to water-timeseries-v2/.venv")

    # Check the package location
    try:
        from water_timeseries.utils import io
        package_path = Path(io.__file__).parent.parent.parent
        print(f"\n📦 water_timeseries package location: {package_path}")

        if "water-timeseries-v2" in str(package_path):
            print("✅ Package is from local editable install")
        else:
            print("⚠️  Package is from cached/installed version")

        # Check for the fix
        import inspect
        source = inspect.getsource(io.save_xarray_dataset)

        if 'consolidated=True' in source:
            print("\n✅✅✅ ZARR FIX IS ACTIVE! ✅✅✅")
            print("   The save_xarray_dataset function uses consolidated=True")
        else:
            print("\n❌ ZARR FIX NOT FOUND")
            print("   The save_xarray_dataset function is still using old code")

            # Show relevant lines
            print("\n   Current to_zarr call:")
            for line in source.split('\n'):
                if 'to_zarr' in line:
                    print(f"     {line.strip()}")

    except ImportError as e:
        print(f"\n❌ Cannot import water_timeseries: {e}")
        print("   Make sure you've run: uv add --editable /path/to/water-timeseries-v2")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()