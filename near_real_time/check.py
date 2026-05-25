import xarray as xr
import os
import glob
import numpy as np
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

# Load environment
load_dotenv()

dynamic_world_dir = os.environ.get('dynamic_world_dir')
split_new_dynamic_world_data_dir = os.environ.get('split_new_dynamic_world_data_dir')

print("=" * 80)
print("COMPARING NETCDF FILE STRUCTURES")
print("=" * 80)

# 1. Find the most recent existing file
all_existing = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
valid_existing = [f for f in all_existing if os.path.getsize(f) > 1024 * 1024]
most_recent = max(valid_existing, key=os.path.getctime) if valid_existing else None

if most_recent:
    print(f"\n1. EXISTING FILE: {os.path.basename(most_recent)}")
    print(f"   Path: {most_recent}")
    print(f"   Size: {os.path.getsize(most_recent) / (1024 ** 3):.2f} GB")

    with xr.open_dataset(most_recent, decode_times=False) as ds:
        print(f"\n   --- Dimensions ---")
        for dim, size in ds.dims.items():
            print(f"     {dim}: {size:,}")

        print(f"\n   --- Coordinates ---")
        for coord in ds.coords:
            print(f"     {coord}: dtype={ds[coord].dtype}, shape={ds[coord].shape}")
            if coord == 'date':
                print(f"        Values (first 5): {ds[coord].values[:5]}")
                print(f"        Values (last 5): {ds[coord].values[-5:]}")
                print(f"        Attributes: {dict(ds[coord].attrs)}")
            elif coord == 'id_geohash':
                print(f"        Sample IDs: {ds[coord].values[:5]}")

        print(f"\n   --- Data Variables ---")
        for var in ds.data_vars:
            print(f"     {var}: dtype={ds[var].dtype}, shape={ds[var].shape}")
            print(f"        Attributes: {dict(ds[var].attrs)}")

        print(f"\n   --- Global Attributes ---")
        for attr, value in ds.attrs.items():
            print(f"     {attr}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
else:
    print("\nNo existing file found!")

# 2. Sample chunk files (first 3, middle 3, last 3)
chunk_files = sorted(glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc")))
print(f"\n{'=' * 80}")
print(f"2. CHUNK FILES (Total: {len(chunk_files)})")
print(f"{'=' * 80}")

# Select sample files
sample_indices = [0, 1, 2, len(chunk_files) // 2 - 1, len(chunk_files) // 2, len(chunk_files) // 2 + 1, -3, -2, -1]
samples = [chunk_files[i] for i in sample_indices if 0 <= i < len(chunk_files)]

for chunk_file in samples:
    print(f"\n--- {os.path.basename(chunk_file)} ---")
    print(f"   Size: {os.path.getsize(chunk_file) / (1024 ** 2):.2f} MB")

    try:
        with xr.open_dataset(chunk_file, decode_times=False) as ds:
            print(f"\n   Dimensions:")
            for dim, size in ds.dims.items():
                print(f"     {dim}: {size:,}")

            print(f"\n   Coordinates:")
            for coord in ds.coords:
                print(f"     {coord}: dtype={ds[coord].dtype}, shape={ds[coord].shape}")
                if coord == 'date':
                    print(f"        Values: {ds[coord].values}")
                    print(f"        Attributes: {dict(ds[coord].attrs)}")
                elif coord == 'id_geohash':
                    print(f"        Sample IDs: {ds[coord].values[:5]}")

            print(f"\n   Data Variables (first 3):")
            for var in list(ds.data_vars)[:3]:
                print(f"     {var}: dtype={ds[var].dtype}, shape={ds[var].shape}")
                print(f"        Attributes: {dict(ds[var].attrs)}")

            if len(ds.data_vars) > 3:
                print(f"     ... and {len(ds.data_vars) - 3} more variables")

            print(f"\n   Global Attributes:")
            for attr, value in ds.attrs.items():
                print(f"     {attr}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")

    except Exception as e:
        print(f"   ERROR reading file: {e}")

print(f"\n{'=' * 80}")
print("3. KEY DIFFERENCES TO CHECK")
print(f"{'=' * 80}")

# Compare specific aspects
if most_recent and chunk_files:
    with xr.open_dataset(most_recent, decode_times=False) as existing:
        with xr.open_dataset(chunk_files[0], decode_times=False) as chunk:

            print("\n--- Date Units Comparison ---")
            print(f"  Existing date units: {existing.date.attrs.get('units', 'NOT FOUND')}")
            print(f"  Chunk date units: {chunk.date.attrs.get('units', 'NOT FOUND')}")

            print("\n--- Variable Names Comparison ---")
            existing_vars = set(existing.data_vars)
            chunk_vars = set(chunk.data_vars)

            common_vars = existing_vars & chunk_vars
            only_existing = existing_vars - chunk_vars
            only_chunk = chunk_vars - existing_vars

            print(f"  Common variables: {sorted(common_vars)}")
            if only_existing:
                print(f"  Only in existing: {sorted(only_existing)}")
            if only_chunk:
                print(f"  Only in chunk: {sorted(only_chunk)}")

            print("\n--- Variable Attributes Comparison (first common variable) ---")
            if common_vars:
                test_var = sorted(common_vars)[0]
                print(f"  Variable: {test_var}")
                print(f"  Existing attributes: {dict(existing[test_var].attrs)}")
                print(f"  Chunk attributes: {dict(chunk[test_var].attrs)}")

            print("\n--- Dimension Order Check ---")
            print(f"  Existing dimensions: {list(existing.dims.keys())}")
            print(f"  Chunk dimensions: {list(chunk.dims.keys())}")

            print("\n--- Data Type Comparison ---")
            print(f"  Existing date dtype: {existing.date.dtype}")
            print(f"  Chunk date dtype: {chunk.date.dtype}")

            print(f"\n  Existing id_geohash dtype: {existing.id_geohash.dtype}")
            print(f"  Chunk id_geohash dtype: {chunk.id_geohash.dtype}")

print(f"\n{'=' * 80}")
print("4. DATE VALUE COMPARISON")
print(f"{'=' * 80}")

if most_recent and chunk_files:
    with xr.open_dataset(most_recent, decode_times=False) as existing:
        existing_dates = existing.date.values
        print(f"\nExisting date range: {existing_dates.min()} to {existing_dates.max()}")
        print(f"Number of existing dates: {len(existing_dates)}")

        # Check a few chunks
        for i, chunk_file in enumerate(chunk_files[:5]):
            with xr.open_dataset(chunk_file, decode_times=False) as chunk:
                original_dates = chunk.date.values
                adjusted_dates = original_dates + 3653
                print(f"\n{os.path.basename(chunk_file)}:")
                print(f"  Original dates: {original_dates}")
                print(f"  Adjusted dates: {adjusted_dates}")
                print(f"  Overlap with existing: {set(adjusted_dates) & set(existing_dates)}")

print(f"\n{'=' * 80}")
print("5. POTENTIAL ISSUES SUMMARY")
print(f"{'=' * 80}")

if most_recent and chunk_files:
    issues = []

    with xr.open_dataset(most_recent, decode_times=False) as existing:
        with xr.open_dataset(chunk_files[0], decode_times=False) as chunk:

            # Check date units
            if existing.date.attrs.get('units') != chunk.date.attrs.get('units'):
                issues.append("❌ Date units don't match")
            else:
                print("✓ Date units match")

            # Check variable names
            if set(existing.data_vars) != set(chunk.data_vars):
                issues.append("❌ Variable names don't match")
            else:
                print("✓ Variable names match")

            # Check dimension order
            if list(existing.dims.keys()) != list(chunk.dims.keys()):
                issues.append("❌ Dimension order differs")
            else:
                print("✓ Dimension order matches")

            # Check id_geohash dtype
            if existing.id_geohash.dtype != chunk.id_geohash.dtype:
                issues.append(f"❌ id_geohash dtype mismatch: {existing.id_geohash.dtype} vs {chunk.id_geohash.dtype}")
            else:
                print(f"✓ id_geohash dtype matches ({existing.id_geohash.dtype})")

            # Check for string vs numeric id_geohash
            if existing.id_geohash.dtype.kind in ['S', 'U']:
                issues.append("⚠️  id_geohash is string type - this may cause memory issues")

            # Check if lakes are unique
            existing_ids = set(existing.id_geohash.values[:1000])  # Sample
            chunk_ids = set(chunk.id_geohash.values)
            overlap_ids = existing_ids & chunk_ids
            if overlap_ids:
                print(f"✓ Sample shows {len(overlap_ids)} overlapping lake IDs (expected)")

    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ No major structural issues found!")

print(f"\n{'=' * 80}")
print("Analysis complete!")
print("=" * 80)