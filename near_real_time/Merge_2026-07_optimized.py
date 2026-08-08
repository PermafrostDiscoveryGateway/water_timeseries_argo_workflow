# %% [markdown]
# ### Noetbook to append 2026-07
# 

# %%
import xarray as xr
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from loguru import logger  # Import the 'logger' instance directly

# Suppress specific warning types
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
# This sets up Loguru to print to the console with a nice format
logger.remove() # Remove default handler
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

# ==========================================
# CONFIGURATION & SWITCH
# ==========================================
RUN_PRODUCTION = True 

dataset_path_large = Path(
    "/Users/helium/Desktop/dynamic_world/lakes_dw_V2d_2016-2026-06_gapfilled_chunked.zarr"
)

data_dir_newData = Path("/Users/helium/Desktop/dynamic_world/output/NRT/NRT_5x2_2026-07_2026-07-31")

OUTPUT_BASENAME = "lakes_dw_V2d_2016-2026-07"
OUTPUT_LOCAL_DIR = Path("/Users/helium/Desktop/dynamic_world/drainage_analysis2/data")
OUTPUT_GS_BUCKET = "gs://pdg-storage-default/workflows_optimization/lake_change_detection"

output_path_local_nc = OUTPUT_LOCAL_DIR / f"{OUTPUT_BASENAME}.nc"
output_path_local_zarr = OUTPUT_LOCAL_DIR / f"{OUTPUT_BASENAME}.zarr"
output_path_gs_zarr = f"{OUTPUT_GS_BUCKET}/{OUTPUT_BASENAME}.zarr"

print(f"Output path (Local): {output_path_local_zarr}")
print(f"Output path (GCS):  {output_path_gs_zarr}")

# %% [markdown]
# ### Load Datasets (Lazy Loading)

# %%
logger.info("Step 1: Opening the large existing Zarr dataset...")
ds_full = xr.open_zarr(dataset_path_large)

# This forces the strings into a standard NumPy format that Zarr handles more predictably
ds_full['id_geohash'] = ds_full['id_geohash'].astype(str)

logger.info(f"Step 2: Scanning for new files in {data_dir_newData}...")
ds_small_flist = list(data_dir_newData.glob("*.nc"))
print(f"Number of new files found: {len(ds_small_flist)}")

logger.info("Step 3: Opening and merging all new files (this may take a moment)...")
# We use combine="nested" and concat_dim="id_geohash" to join the tiles side-by-side
ds_new_merged = xr.open_mfdataset(ds_small_flist, combine="nested", concat_dim="id_geohash")

# %% [markdown]
# ### Execution Block (Test or Production)

# %%
if not RUN_PRODUCTION:
    print("--- RUNNING TEST SUBSET ---")
    
    test_lakes = ds_full.id_geohash.values[:10_000] 
    ds_old_test = ds_full.sel(id_geohash=test_lakes)

    # Select by Identity (Corrected)
    ds_new_test = ds_new_merged.sel(id_geohash=test_lakes)

    ds_new_test_aligned = ds_new_test.reindex(id_geohash=ds_old_test.id_geohash)
    ds_combined_test = xr.concat([ds_old_test, ds_new_test_aligned], dim="date")

    print(f"Old Data - Lakes: {ds_old_test.id_geohash.size}, Months: {ds_old_test.date.size}")
    print(f"New Month - Lakes: {ds_new_test_aligned.id_geohash.size}, Months: {ds_new_test_aligned.date.size}")
    print(f"Combined - Lakes: {ds_combined_test.id_geohash.size}, Months: {ds_combined_test.date.size}")

    # --- TEST VALIDATION ---
    new_date = ds_combined_test.date.values[-1]
    water_slice = ds_combined_test["water"].sel(date=new_date) 
    non_null_count = water_slice.count().values
    print(f"Validation for {new_date}:")
    print(f"  - Non-Null 'water' values: {non_null_count} / {ds_combined_test.id_geohash.size}")

    # FIX: Added align_chunks=True to resolve the Dask chunk overlap error
    print(f"Saving test results to {output_path_local_zarr}...")
    ds_combined_test.to_zarr(output_path_local_zarr, align_chunks=True, mode='w')
    print("Test complete.")

else:
    print("--- RUNNING PRODUCTION UPDATE ---")
    
    logger.info("Step 4: Aligning new data coordinates to the master lake list...")
    # This ensures that every lake in your master store has a corresponding slot for July 2026.
    ds_new_aligned = ds_new_merged.reindex(id_geohash=ds_full.id_geohash)

    logger.info("Step 5: Concatenating the old history with the new month...")
    # Concatenate the full datasets along the date dimension (Lazy operation)
    ds_combined = xr.concat([ds_full, ds_new_aligned], dim="date")

    logger.info(f"Step 6: Saving final result to local storage ({output_path_local_zarr})...")
    # Using mode='w' as per your latest script version (overwrites the file with combined data)
    ds_combined.to_zarr(output_path_local_zarr, mode='w', align_chunks=True)

    logger.info(f"Step 7: Uploading final result to Google Cloud Storage ({output_path_gs_zarr})...")
    # Note: Ensure your environment has gcloud auth configured or the appropriate credentials set.
    ds_combined.to_zarr(output_path_gs_zarr, mode='w', align_chunks=True)

    # --- PRODUCTION VALIDATION ---
    logger.info("Step 8: Running final validation checks...")
    new_date = ds_combined.date.values[-1]
    water_slice = ds_combined["water"].sel(date=new_date) 
    non_null_count = water_slice.count().values
    print(f"Validation for {new_date}:")
    print(f"  - Total Lakes: {ds_combined.id_geohash.size}")
    print(f"  - Non-Null 'water' values: {non_null_count}")
    print(f"  - Coverage: {(non_null_count / ds_combined.id_geohash.size) * 100:.2f}%")

    logger.success("Production update complete!")
