from water_timeseries.breakpoint import NRTBreakpoint
from water_timeseries.dataset import DWDataset
import datetime
import pandas as pd
import os
import xarray as xr
import geopandas as gpd
import warnings

warnings.filterwarnings("ignore")

# Set environment variable to handle Zarr threading issues in debug mode
os.environ["ZARR_V3_EXPERIMENTAL_API"] = "0"

# File paths
water_dataset_file = '/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/dw_downloads/dynamic_world_data_v2.zarr'
vector_dataset_file = '/Users/helium/ncsa/pdg/water-timeseries-v2/tests/data/lake_polygons.parquet'

# Verify files exist
print(f"Water dataset exists: {os.path.exists(water_dataset_file)}")
print(f"Vector dataset exists: {os.path.exists(vector_dataset_file)}")

# Step 1: Load the water data
print("\n" + "=" * 60)
print("STEP 1: Loading water dataset")
print("=" * 60)

water_ds = None
try:
    water_ds = xr.open_zarr(water_dataset_file, consolidated=False)
    print("✓ Successfully loaded dataset")
except Exception as e:
    print(f"Error loading: {e}")
    raise

print(f"\nDataset loaded!")
print(f"Dimensions: {water_ds.dims}")
print(f"Variables: {list(water_ds.data_vars)[:5]}...")
print(f"Coordinates: {list(water_ds.coords)}")

# Step 2: Load vector dataset
print("\n" + "=" * 60)
print("STEP 2: Loading vector dataset")
print("=" * 60)

try:
    lake_gdf = gpd.read_parquet(vector_dataset_file)
    print(f"Loaded {len(lake_gdf)} lake polygons")
except Exception as e:
    print(f"Error loading vector dataset: {e}")
    lake_gdf = None

# Step 3: Create DWDataset
print("\n" + "=" * 60)
print("STEP 3: Creating DWDataset")
print("=" * 60)

dataset = DWDataset(ds=water_ds, mask_data=False)
print(f"✓ Dataset created successfully")
print(f"  Number of lakes: {len(dataset.object_ids_)}")
print(f"  Date range: {dataset.dates_[0]} to {dataset.dates_[-1]}")
print(f"  Water column: {dataset.water_column}")

# Step 4: Initialize NRTBreakpoint
print("\n" + "=" * 60)
print("STEP 4: Initializing NRTBreakpoint")
print("=" * 60)

nrt_breakpoint = NRTBreakpoint(kwargs_break={})
print("✓ NRTBreakpoint initialized")

# Step 5: Choose analysis date - AUTOMATICALLY USE LATEST AVAILABLE DATE
print("\n" + "=" * 60)
print("STEP 5: Setting Analysis Date")
print("=" * 60)

# Convert dates to pandas datetime for proper handling
dates_clean = []
for d in dataset.dates_:
    if isinstance(d, str):
        dates_clean.append(pd.to_datetime(d))
    else:
        dates_clean.append(d)

# Get the most recent date
analysis_date = max(dates_clean)
print(f"Most recent date in dataset: {analysis_date}")
print(f"Analysis date (YYYY-MM): {analysis_date.strftime('%Y-%m')}")

# Optional: If you want a specific date, uncomment and modify below:
# specific_date = pd.to_datetime("2024-06-01")
# if specific_date in dates_clean:
#     analysis_date = specific_date
#     print(f"Using specified date: {analysis_date}")
# else:
#     print(f"Specified date not found, using most recent: {analysis_date}")

# Step 6: Run NRT breakpoint detection
print("\n" + "=" * 60)
print("STEP 6: Running NRT Breakpoint Detection")
print("=" * 60)

# Process first 5 lakes
sample_lake_ids = dataset.object_ids_[:min(5, len(dataset.object_ids_))]
print(f"Processing {len(sample_lake_ids)} lakes: {sample_lake_ids}")

try:
    results = nrt_breakpoint.calculate_break(
        dataset=dataset,
        analysis_date=analysis_date,  # Now using valid date
        data_aggregation_period="all",
        object_id=sample_lake_ids
    )

    print(f"\n✓ Results shape: {results.shape}")
    print("\nResults columns:")
    print(results.columns.tolist())
    print("\nFirst few results:")
    print(results.head(10))

except Exception as e:
    print(f"Error running NRT: {e}")
    import traceback

    traceback.print_exc()
    results = None

# Step 7: Analyze results
print("\n" + "=" * 60)
print("STEP 7: Analyzing Results")
print("=" * 60)

if results is not None and len(results) > 0:
    if 'water_observed' in results.columns and 'water_predicted' in results.columns:
        results['water_diff'] = results['water_observed'] - results['water_predicted']
        results['diff_percent'] = (results['water_diff'] / results['water_predicted']) * 100

        # Look for significant negative differences (potential breakpoints)
        significant = results[results['water_diff'] < -0.15]  # 15% below prediction

        print(f"\nLakes with potential breakpoints: {len(significant)}")
        if len(significant) > 0:
            print("\nPotential breakpoints:")
            print(significant[['water_observed', 'water_predicted', 'water_diff', 'diff_percent']])
        else:
            print("No significant breakpoints detected in sample")

        # Save results
        output_file = f'nrt_results_{analysis_date.strftime("%Y%m")}.csv'
        results.to_csv(output_file)
        print(f"\nResults saved to: {output_file}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)

# Additional: Show available dates
print("\n" + "=" * 60)
print("AVAILABLE DATES IN DATASET")
print("=" * 60)

unique_years = sorted(set([d.year for d in dates_clean]))
unique_months = sorted(set([d.month for d in dates_clean]))

print(f"Years available: {unique_years}")
print(f"Months available: {unique_months}")
print(f"Total dates: {len(dates_clean)}")

# Show first and last few dates
print(f"\nFirst 5 dates: {[d.strftime('%Y-%m') for d in dates_clean[:5]]}")
print(f"Last 5 dates: {[d.strftime('%Y-%m') for d in dates_clean[-5:]]}")