#!/usr/bin/env python3
import xarray as xr
import numpy as np
import pandas as pd

# Load your dataset
ds = xr.open_zarr("/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/dw_downloads/dynamic_world_data_v2.zarr")

print("=" * 60)
print("DATASET DIAGNOSTICS")
print("=" * 60)

print(f"\nDimensions:")
print(f"  Lakes: {ds.dims['id_geohash']}")
print(f"  Time points: {ds.dims['date']}")

# Check water variable
water = ds['water']
print(f"\nWater data shape: {water.shape}")
print(f"Total cells: {water.size}")

# Count valid (non-NaN) values
valid_mask = ~np.isnan(water.values)
valid_count = valid_mask.sum()
total_count = water.size
print(f"\nValid (non-NaN) values: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")

# Check each lake's valid time points
valid_per_lake = (~np.isnan(water.values)).sum(axis=1)
print(f"\nValid time points per lake:")
print(f"  Min: {valid_per_lake.min()}")
print(f"  Max: {valid_per_lake.max()}")
print(f"  Mean: {valid_per_lake.mean():.1f}")
print(f"  Median: {np.median(valid_per_lake)}")

# Count lakes with enough data (need at least 3-5 points for simple method)
lakes_with_enough = (valid_per_lake >= 3).sum()
print(f"\nLakes with >=3 valid points: {lakes_with_enough}/{len(ds.id_geohash)}")
lakes_with_enough_5 = (valid_per_lake >= 5).sum()
print(f"Lakes with >=5 valid points: {lakes_with_enough_5}/{len(ds.id_geohash)}")

# Check time range
if 'date' in ds.coords:
    dates = ds.date.values
    print(f"\nTime range:")
    print(f"  First date: {dates[0]}")
    print(f"  Last date: {dates[-1]}")
    print(f"  Total span: {len(dates)} time points")

# Check a sample of actual water area values
print(f"\nSample water area values (first 5 lakes, first 10 time points):")
for i in range(min(5, len(ds.id_geohash))):
    sample = water[i, :10].values
    non_nan_sample = sample[~np.isnan(sample)]
    if len(non_nan_sample) > 0:
        print(f"  Lake {i}: {non_nan_sample[:5]}... (range: {np.nanmin(sample):.2f} to {np.nanmax(sample):.2f})")
    else:
        print(f"  Lake {i}: All NaN in first 10 points")

# Check if there's any variation in the data
print(f"\nVariation check:")
all_values = water.values[~np.isnan(water.values)]
if len(all_values) > 0:
    print(f"  Overall min: {all_values.min():.2f}")
    print(f"  Overall max: {all_values.max():.2f}")
    print(f"  Overall mean: {all_values.mean():.2f}")
    print(f"  Std deviation: {all_values.std():.2f}")
    if all_values.std() < 0.01:
        print(f"  ⚠️ Very low variation - likely constant values!")
else:
    print(f"  ⚠️ No valid numeric values found!")

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

if valid_count == 0:
    print("❌ No valid data found! The water variable is all NaN.")
    print("   Check if the dataset was created correctly.")
elif valid_per_lake.max() < 3:
    print("❌ Not enough time points per lake (need at least 3-5).")
    print("   Your time series is too short for breakpoint detection.")
elif lakes_with_enough == 0:
    print("❌ No lake has enough valid data points.")
    print("   Consider using a dataset with longer time series.")
else:
    print(f"✓ Found {lakes_with_enough} lakes with sufficient data.")
    print(f"  But the 'simple' method detected 0 breakpoints.")
    print(f"  Possible reasons:")
    print(f"    1. Water areas are stable (no significant changes)")
    print(f"    2. Detection thresholds are too strict")
    print(f"    3. Need to adjust method parameters")