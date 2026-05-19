import xarray as xr
import pandas as pd

# Load your test file
test_file = "/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/data/dynamic_world_test/lakes_dw_V2d_test_quality.nc"
ds = xr.open_dataset(test_file)

print("=== Test File Info ===")
print(f"Number of lakes: {len(ds.id_geohash)}")
print(f"Date range: {ds.date.values[0]} to {ds.date.values[-1]}")
print(f"Number of dates: {len(ds.date.values)}")

# Check for June 2024 data
target_date = pd.to_datetime("2024-06-01")
print(f"\n=== Checking for {target_date} ===")

if target_date in ds.date.values:
    print(f"✅ {target_date} exists in the file")

    # Check how many lakes have data for this date
    june_data = ds.sel(date=target_date)
    valid_lakes = june_data.dropna(dim="id_geohash", how="all")
    print(f"  Lakes with data on {target_date}: {len(valid_lakes.id_geohash)}")
else:
    print(f"❌ {target_date} NOT in the file")
    print(f"Closest dates: {ds.date.values[0]} to {ds.date.values[-1]}")

    # Find what June dates are available
    june_dates = [d for d in ds.date.values if pd.to_datetime(d).month == 6]
    print(f"June dates in file: {june_dates[:10]}...")

# Check historical data for a sample lake
sample_lake = ds.id_geohash.values[0]
lake_data = ds.sel(id_geohash=sample_lake)
print(f"\n=== Sample Lake {sample_lake} ===")
print(f"Total observations: {len(lake_data.date)}")
print(f"Non-NaN water values: {lake_data.water.count().item()}")
print(f"Date range: {lake_data.date.values[0]} to {lake_data.date.values[-1]}")