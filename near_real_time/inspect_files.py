#!/usr/bin/env python3
import xarray as xr
import pandas as pd
import numpy as np
import os
import glob
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Get directories from environment
dynamic_world_dir = os.environ.get('dynamic_world_dir')
split_new_dynamic_world_data_dir = os.environ.get('split_new_dynamic_world_data_dir')

if not dynamic_world_dir or not split_new_dynamic_world_data_dir:
    print("ERROR: Could not find environment variables. Make sure .env file is loaded.")
    print(f"dynamic_world_dir: {dynamic_world_dir}")
    print(f"split_new_dynamic_world_data_dir: {split_new_dynamic_world_data_dir}")
    exit(1)

print("=" * 80)
print("INSPECTING DYNAMIC WORLD FILES")
print("=" * 80)

# 1. Find the most recent existing file
all_existing_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
valid_existing = [f for f in all_existing_files if os.path.getsize(f) > 1024 * 1024]

if not valid_existing:
    print("No valid existing files found!")
else:
    most_recent = max(valid_existing, key=os.path.getctime)
    print(f"\n1. Most recent existing file: {os.path.basename(most_recent)}")
    print(f"   Full path: {most_recent}")
    print(f"   File size: {os.path.getsize(most_recent) / (1024 * 1024):.2f} MB")

    # Open without decoding
    print("\n   --- Opening with decode_times=False ---")
    ds_existing = xr.open_dataset(most_recent, decode_times=False)

    print(f"   Dimensions: {dict(ds_existing.dims)}")
    print(f"   Variables: {list(ds_existing.data_vars.keys())}")

    if 'date' in ds_existing.variables:
        date_vals = ds_existing.date.values
        print(f"\n   DATE VARIABLE:")
        print(f"     Shape: {date_vals.shape}")
        print(f"     Dtype: {date_vals.dtype}")
        print(f"     First 10 values: {date_vals[:10]}")
        print(f"     Last 10 values: {date_vals[-10:]}")
        print(f"     Min value: {date_vals.min()}")
        print(f"     Max value: {date_vals.max()}")
        print(f"     Mean value: {date_vals.mean():.2f}")
        print(f"     Any negative? {(date_vals < 0).any()}")
        print(f"     Any huge (>1e10)? {(date_vals > 1e10).any()}")
        print(f"     Any NaN? {np.isnan(date_vals).any()}")

        # Check attributes
        print(f"\n     Attributes:")
        for attr, value in ds_existing.date.attrs.items():
            print(f"       {attr}: {value}")

        # Try manual conversion to see what dates would be
        units = ds_existing.date.attrs.get('units', '')
        if 'days since' in units:
            import re

            match = re.search(r'days since (\d{4}-\d{2}-\d{2})', units)
            if match:
                ref_date = pd.Timestamp(match.group(1))
                print(f"\n     Reference date: {ref_date}")
                print(f"     Would convert to:")
                for i, val in enumerate(date_vals[:5]):
                    converted = ref_date + pd.Timedelta(days=int(val))
                    print(f"       {val} days -> {converted.date()}")

    # Now try opening with decode_times=True
    print("\n   --- Opening with decode_times=True ---")
    try:
        ds_existing_decoded = xr.open_dataset(most_recent, decode_times=True)
        if 'date' in ds_existing_decoded.variables:
            decoded_dates = ds_existing_decoded.date.values
            print(f"     Success! Decoded dates shape: {decoded_dates.shape}")
            print(f"     First 5 decoded dates: {decoded_dates[:5]}")
            print(f"     Date range: {decoded_dates.min()} to {decoded_dates.max()}")
        ds_existing_decoded.close()
    except Exception as e:
        print(f"     ERROR decoding dates: {e}")

    ds_existing.close()

print("\n" + "=" * 80)
print("INSPECTING NEW CHUNK FILES")
print("=" * 80)

# 2. Inspect chunk files
chunk_files = glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc"))
print(f"\nFound {len(chunk_files)} chunk files")

if chunk_files:
    # Check first few chunk files
    for i, chunk_file in enumerate(chunk_files[:5]):  # Check first 5 files
        print(f"\n--- Chunk file {i + 1}: {os.path.basename(chunk_file)} ---")
        print(f"   File size: {os.path.getsize(chunk_file) / (1024 * 1024):.2f} MB")

        try:
            ds_chunk = xr.open_dataset(chunk_file, decode_times=False)

            print(f"   Dimensions: {dict(ds_chunk.dims)}")

            if 'date' in ds_chunk.variables:
                date_vals = ds_chunk.date.values
                print(f"\n   DATE VARIABLE:")
                print(f"     Shape: {date_vals.shape}")
                print(f"     Dtype: {date_vals.dtype}")
                print(f"     Unique dates: {len(np.unique(date_vals))}")
                print(f"     First 10 values: {date_vals[:10]}")
                print(f"     Min value: {date_vals.min()}")
                print(f"     Max value: {date_vals.max()}")
                print(f"     Any negative? {(date_vals < 0).any()}")
                print(f"     Any huge (>1e10)? {(date_vals > 1e10).any()}")
                print(f"     Any NaN? {np.isnan(date_vals).any()}")

                # Check for overflow values like -9223372036854775806
                overflow_val = -9223372036854775806
                if (date_vals == overflow_val).any():
                    count = (date_vals == overflow_val).sum()
                    print(f"     WARNING: Found {count} overflow values ({overflow_val})!")

                print(f"\n     Attributes:")
                for attr, value in ds_chunk.date.attrs.items():
                    print(f"       {attr}: {value}")

                # Try manual conversion
                units = ds_chunk.date.attrs.get('units', '')
                if 'days since' in units:
                    import re

                    match = re.search(r'days since (\d{4}-\d{2}-\d{2})', units)
                    if match:
                        ref_date = pd.Timestamp(match.group(1))
                        print(f"\n     Reference date: {ref_date}")
                        # Try converting first few valid values
                        valid_vals = date_vals[date_vals > -1e10]  # Filter out overflow
                        if len(valid_vals) > 0:
                            print(f"     Sample conversion (first 3 valid values):")
                            for val in valid_vals[:3]:
                                try:
                                    converted = ref_date + pd.Timedelta(days=int(val))
                                    print(f"       {val} days -> {converted.date()}")
                                except Exception as e:
                                    print(f"       {val} days -> ERROR: {e}")

            ds_chunk.close()

        except Exception as e:
            print(f"   ERROR opening file: {e}")

print("\n" + "=" * 80)
print("COMPARING DATE RANGES")
print("=" * 80)

# Compare date ranges between existing and new files
if valid_existing and chunk_files:
    # Get existing date range
    ds_exist = xr.open_dataset(most_recent, decode_times=False)
    exist_dates = ds_exist.date.values

    # Get date range from first valid chunk
    first_chunk = chunk_files[0]
    ds_chunk = xr.open_dataset(first_chunk, decode_times=False)
    chunk_dates = ds_chunk.date.values

    print("\nExisting file date range:")
    print(f"  Min: {exist_dates.min()}")
    print(f"  Max: {exist_dates.max()}")
    print(f"  Units: {ds_exist.date.attrs.get('units', 'unknown')}")

    print("\nChunk file date range:")
    print(f"  Min: {chunk_dates.min()}")
    print(f"  Max: {chunk_dates.max()}")
    print(f"  Units: {ds_chunk.date.attrs.get('units', 'unknown')}")

    # Check if they use same reference date
    exist_units = ds_exist.date.attrs.get('units', '')
    chunk_units = ds_chunk.date.attrs.get('units', '')

    if exist_units != chunk_units:
        print("\n⚠️  WARNING: Different unit definitions!")
        print(f"  Existing: {exist_units}")
        print(f"  Chunk: {chunk_units}")

        # Extract reference dates
        import re

        exist_match = re.search(r'days since (\d{4}-\d{2}-\d{2})', exist_units)
        chunk_match = re.search(r'days since (\d{4}-\d{2}-\d{2})', chunk_units)

        if exist_match and chunk_match:
            exist_ref = pd.Timestamp(exist_match.group(1))
            chunk_ref = pd.Timestamp(chunk_match.group(1))
            print(f"\n  Existing reference date: {exist_ref}")
            print(f"  Chunk reference date: {chunk_ref}")

            if exist_ref != chunk_ref:
                print(f"  ⚠️  Different reference dates! Offset of {(chunk_ref - exist_ref).days} days")

                # Show actual dates they represent
                print("\n  Actual dates represented (example):")
                for val in [365, 730, 1095]:  # 1,2,3 years approx
                    if val < len(exist_dates):
                        exist_actual = exist_ref + pd.Timedelta(days=int(val))
                        chunk_actual = chunk_ref + pd.Timedelta(days=int(val))
                        print(f"    Value {val}: Existing = {exist_actual.date()}, Chunk = {chunk_actual.date()}")

    ds_exist.close()
    ds_chunk.close()

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

# Provide recommendations based on what we found
if chunk_files:
    # Check for overflow in first chunk
    ds_test = xr.open_dataset(chunk_files[0], decode_times=False)
    test_dates = ds_test.date.values
    overflow_val = -9223372036854775806

    if (test_dates == overflow_val).any():
        print("\n⚠️  CRITICAL: Found overflow values in chunk files!")
        print("   These are likely from partial/corrupted downloads or incomplete writes.")
        print("   Recommendation: Regenerate the chunk files or filter out corrupted ones.")

    # Check for different reference dates
    exist_units = ds_exist.date.attrs.get('units', '') if 'exist_units' in locals() else ''
    chunk_units = ds_chunk.date.attrs.get('units', '') if 'chunk_units' in locals() else ''

    if exist_units and chunk_units and exist_units != chunk_units:
        print("\n⚠️  Unit mismatch between existing and new files!")
        print("   You need to convert dates to a common reference before merging.")
        print("   Use the script with proper date conversion as shown earlier.")

    ds_test.close()

print("\n" + "=" * 80)
print("To save this output to a file: python inspect_files.py > inspection_report.txt")
print("=" * 80)