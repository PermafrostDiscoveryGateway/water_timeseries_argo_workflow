import xarray as xr
import os
import glob
import numpy as np
from dotenv import load_dotenv

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

# Open files once
old = xr.open_dataset(existing_file, decode_times=False)
new = xr.open_dataset(new_file, decode_times=False)

print("\n--- ENCODING (Compression Settings) ---")
print(f"Old file encoding: {old.bare.encoding}")
print(f"New file encoding: {new.bare.encoding}")

print("\n--- EXISTING FILE ---")
print(f"File: {os.path.basename(existing_file)}")
print(f"  Size: {os.path.getsize(existing_file) / (1024 ** 3):.2f} GB")
print(f"  Lakes: {len(old.id_geohash):,}")
print(f"  Dates: {len(old.date)}")
print(f"  Date range: {old.date.values.min()} to {old.date.values.max()}")
print(f"  Variables: {list(old.data_vars.keys())}")

# Check data density for existing
for var in list(old.data_vars.keys())[:1]:
    data = old[var].values
    nan_count = np.isnan(data).sum()
    print(f"  {var} NaN %: {nan_count / data.size * 100:.1f}%")

print("\n--- NEW FILE ---")
print(f"File: {os.path.basename(new_file)}")
print(f"  Size: {os.path.getsize(new_file) / (1024 ** 3):.2f} GB")
print(f"  Lakes: {len(new.id_geohash):,}")
print(f"  Dates: {len(new.date)}")
print(f"  Date range: {new.date.values.min()} to {new.date.values.max()}")
print(f"  Variables: {list(new.data_vars.keys())}")

# Check data density for new
for var in list(new.data_vars.keys())[:1]:
    data = new[var].values
    nan_count = np.isnan(data).sum()
    print(f"  {var} NaN %: {nan_count / data.size * 100:.1f}%")

# Check lake ID overlap
existing_ids_sample = set(old.id_geohash.values[:1000])
new_ids_sample = set(new.id_geohash.values[:1000])
overlap = existing_ids_sample & new_ids_sample
print(f"\n--- LAKE ID OVERLAP ---")
print(f"  Lake ID overlap (first 1000): {len(overlap)}/1000")

# Compare a few specific lakes
print("\n--- DATA INTEGRITY CHECK (Sample Lakes) ---")
sample_lakes = list(old.id_geohash.values)[:5]

for lake_id in sample_lakes:
    print(f"\n  Lake: {lake_id}")

    # Get indices
    existing_idx = list(old.id_geohash.values).index(lake_id)
    new_idx = list(new.id_geohash.values).index(lake_id)

    # Compare first few dates
    for date_idx in range(min(5, len(old.date))):
        existing_val = old.bare.values[existing_idx, date_idx]
        new_val = new.bare.values[new_idx, date_idx]

        # Check if values match
        if np.isnan(existing_val) and np.isnan(new_val):
            match = "✓"
        elif not np.isnan(existing_val) and not np.isnan(new_val) and abs(existing_val - new_val) < 0.01:
            match = "✓"
        else:
            match = "✗"

        print(f"    Date {old.date.values[date_idx]}: existing={existing_val:.2f}, new={new_val:.2f} {match}")

print("\n" + "=" * 60)
print("✓ MERGE VERIFICATION COMPLETE!")
print("=" * 60)

# Close files
old.close()
new.close()