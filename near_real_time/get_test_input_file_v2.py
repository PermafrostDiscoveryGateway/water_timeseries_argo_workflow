import xarray as xr
import numpy as np
import random
import pandas as pd
from pathlib import Path


def extract_smart_test_subset(
        input_nc_file: str | Path,
        output_nc_file: str | Path,
        num_lakes: int = 10000,
        min_historical_points: int = 10,  # Require at least 10 historical points
        analysis_date: str = "2025-06-01"
) -> Path:
    """
    Extract a test subset that actually has good data.
    """
    input_nc_file = Path(input_nc_file)
    output_nc_file = Path(output_nc_file)

    print(f"Opening {input_nc_file}...")
    ds = xr.open_dataset(input_nc_file)

    analysis_date_ts = pd.to_datetime(analysis_date)

    # Check which lakes have good historical data
    print("Checking which lakes have sufficient historical data...")
    valid_lakes = []

    # Sample a larger pool first
    all_lake_ids = ds.id_geohash.values
    sample_size = min(50000, len(all_lake_ids))
    candidate_lakes = np.random.choice(all_lake_ids, sample_size, replace=False)

    for i, lake_id in enumerate(candidate_lakes):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(candidate_lakes)} lakes...")

        # Get historical data for this lake (before analysis_date)
        lake_data = ds.sel(id_geohash=lake_id)
        historical = lake_data.where(lake_data.date < analysis_date_ts, drop=True)

        # Count non-NaN water values
        water_vals = historical.water.values
        n_valid = np.sum(~np.isnan(water_vals))

        if n_valid >= min_historical_points:
            valid_lakes.append(lake_id)

        if len(valid_lakes) >= num_lakes:
            break

    print(f"Found {len(valid_lakes)} lakes with >= {min_historical_points} historical points")

    if len(valid_lakes) == 0:
        print("No lakes found with sufficient data! Try lowering min_historical_points")
        return None

    # Extract the valid lakes
    print(f"Extracting {len(valid_lakes)} lakes...")
    ds_subset = ds.sel(id_geohash=valid_lakes[:num_lakes])

    # Save to new file
    print(f"Saving to {output_nc_file}...")
    ds_subset.to_netcdf(output_nc_file)

    # Get file size
    subset_size_mb = output_nc_file.stat().st_size / (1024 ** 2)
    print(f"\n✅ Created test file: {subset_size_mb:.2f} MB")
    print(f"   Lakes: {len(valid_lakes[:num_lakes])}")
    print(f"   Min historical points: {min_historical_points}")

    ds.close()
    ds_subset.close()

    return output_nc_file


# Alternative: Extract lakes that have data for a specific month
def extract_monthly_test_subset(
        input_nc_file: str | Path,
        output_nc_file: str | Path,
        target_month: int = 6,  # June
        years_required: int = 3,  # Need at least 3 years of data
        num_lakes: int = 5000
) -> Path:
    """
    Extract lakes that have good June data over multiple years.
    """
    input_nc_file = Path(input_nc_file)
    output_nc_file = Path(output_nc_file)

    print(f"Opening {input_nc_file}...")
    ds = xr.open_dataset(input_nc_file)

    all_lake_ids = ds.id_geohash.values
    valid_lakes = []

    print(f"Checking lakes for June data over {years_required}+ years...")

    for i, lake_id in enumerate(all_lake_ids):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(all_lake_ids)} lakes...")

        # Get June data only
        lake_data = ds.sel(id_geohash=lake_id)
        june_data = lake_data.where(lake_data.date.dt.month == target_month, drop=True)
        water_vals = june_data.water.values
        n_valid = np.sum(~np.isnan(water_vals))

        if n_valid >= years_required:
            valid_lakes.append(lake_id)

        if len(valid_lakes) >= num_lakes:
            break

    print(f"Found {len(valid_lakes)} lakes with {years_required}+ years of June data")

    if len(valid_lakes) == 0:
        print("No lakes found! Try reducing years_required")
        return None

    # Extract subset
    ds_subset = ds.sel(id_geohash=valid_lakes[:num_lakes])
    ds_subset.to_netcdf(output_nc_file)

    print(f"✅ Created test file with {len(valid_lakes[:num_lakes])} lakes")
    print(f"   Each lake has at least {years_required} June observations")

    ds.close()
    ds_subset.close()

    return output_nc_file


# Usage:
if __name__ == "__main__":
    # Method 1: Smart extraction based on data quality
    extract_smart_test_subset(
        input_nc_file="/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/data/dynamic_world/lakes_dw_V2d.nc",
        output_nc_file="/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/data/dynamic_world_test/lakes_dw_V2d_test_quality.nc",
        num_lakes=10000,
        min_historical_points=10
    )

    # Method 2: Specifically for monthly analysis
    extract_monthly_test_subset(
        input_nc_file="/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/data/dynamic_world/lakes_dw_V2d.nc",
        output_nc_file="/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/data/dynamic_world_test/lakes_dw_V2d_test_june.nc",
        target_month=6,
        years_required=3,
        num_lakes=5000
    )