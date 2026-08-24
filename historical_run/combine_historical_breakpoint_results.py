import nest_asyncio
# Zarr v3's async I/O keeps a per-thread event loop; running under a debugger
# (e.g. PyCharm's pydevd) can hand a later store-open a Future tied to a stale
# loop ("Task ... got Future ... attached to a different loop"). Patching
# asyncio to tolerate re-entrant loops here avoids that crash.
nest_asyncio.apply()

from dotenv import load_dotenv
import os
import glob
import sys
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
import xarray as xr

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.region_boundaries import get_region_boundaries
from historical_run.process_historical_BST import generate_date_range_from_env

ALL_REGIONS = ['ALASKA', 'CANADA1', 'CANADA2', 'CANADA3', 'CANADA4', 'EURASIA1', 'EURASIA2', 'EURASIA3']


def merge_regions_for_date(date, output_dir, regions):
    """Open and merge every region's breakpoint zarr for a single date along id_geohash, skipping missing stores."""
    results_paths_to_merge = []
    for region in regions:
        path = os.path.join(output_dir, region, 'breakpoint_zarr', f"breakpoints_{date}.zarr")
        if not os.path.exists(path):
            logger.warning(f"Breakpoint zarr does not exist, skipping: {path}")
            continue
        results_paths_to_merge.append(path)

    if not results_paths_to_merge:
        logger.error(f"No breakpoint zarr datasets available to merge for {date}")
        return None

    logger.info(f"Opening {len(results_paths_to_merge)} region breakpoint zarr datasets for {date}...")
    merged = xr.open_mfdataset(
        results_paths_to_merge, engine="zarr", combine="nested", concat_dim="id_geohash"
    )

    # Region boundary boxes can overlap (e.g. ALASKA/CANADA1), so the same lake may be
    # processed by more than one region and appear more than once here. Keep the first
    # occurrence (in region order) and drop the rest so id_geohash stays unique for alignment.
    _, unique_index = np.unique(merged["id_geohash"].values, return_index=True)
    if len(unique_index) < merged.sizes["id_geohash"]:
        n_dupes = merged.sizes["id_geohash"] - len(unique_index)
        logger.warning(
            f"Dropping {n_dupes} duplicate id_geohash entries (overlapping region boundaries) for {date}"
        )
        merged = merged.isel(id_geohash=np.sort(unique_index))
        # Dropping scattered positions above fragments the dask chunks into ragged,
        # non-uniform sizes, which zarr refuses to write. Consolidate back to one chunk.
        merged = merged.chunk({"id_geohash": -1})

    return merged


def main():
    logger.debug("Combining regional historical breakpoint zarr datasets into a single archive")
    env_path = None
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    output_dir = os.environ['output_dir']
    combined_zarr_datasets = os.environ.get('combined_zarr_datasets', output_dir)
    os.makedirs(combined_zarr_datasets, exist_ok=True)

    start_year = int(os.environ.get("start_year", 2015))
    start_month = int(os.environ.get("start_month", 6))
    end_year = int(os.environ.get("end_year", 2026))
    end_month = int(os.environ.get("end_month", 9))
    months_str = os.environ.get("months_to_process", "6,7,8,9")
    months = [int(m.strip()) for m in months_str.split(",")]

    dates_to_process = generate_date_range_from_env(
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        months=months
    )

    if not dates_to_process:
        logger.error("No dates generated for the given start/end/months configuration - aborting")
        return {'success': False, 'error': 'No dates to process'}

    regions = list(get_region_boundaries().keys())
    regions = [r for r in ALL_REGIONS if r in regions]

    logger.info(f"Combining {len(dates_to_process)} date(s) across {len(regions)} region(s)")
    logger.info(f"Date range: {dates_to_process[0]} to {dates_to_process[-1]}")

    per_date_items = [
        (date, ds) for date, ds in (
            (date, merge_regions_for_date(date, output_dir, regions)) for date in dates_to_process
        )
        if ds is not None
    ]

    if not per_date_items:
        logger.error("No breakpoint zarr datasets available to merge - aborting")
        return {'success': False, 'error': 'No breakpoint zarr datasets found'}

    # Each per-region store already has a per-lake 'date' variable (breakpoint date),
    # which collides with the 'date' dimension we're about to create to stack months.
    # Rename it out of the way before adding the real month coordinate.
    per_date_datasets = [
        ds.rename({"date": "breakpoint_date"})
        .assign_coords(date=pd.Timestamp(f"{date}-01"))
        .expand_dims("date")
        for date, ds in per_date_items
    ]

    if len(per_date_datasets) > 1:
        ds_combined = xr.concat(per_date_datasets, dim="date", join="outer")
    else:
        ds_combined = per_date_datasets[0]

    # Concatenation/alignment can leave ragged dask chunks (especially with join="outer"
    # padding mismatched id_geohash coordinates), which zarr refuses to write. Consolidate
    # to a single chunk per dimension before saving.
    ds_combined = ds_combined.chunk({"id_geohash": -1, "date": -1})

    start_date_str = f"{start_year}-{start_month:02d}"
    end_date_str = f"{end_year}-{end_month:02d}"
    new_zarr_dataset_name = f"historical_breakpoint_results_{start_date_str}_{end_date_str}.zarr"
    new_zarr_dataset_path = os.path.join(combined_zarr_datasets, new_zarr_dataset_name)

    logger.info(f"Saving combined result to {new_zarr_dataset_path}")
    ds_combined.to_zarr(new_zarr_dataset_path, mode='w', align_chunks=True)

    logger.success(f"Combined historical zarr dataset written to {new_zarr_dataset_path}")
    return {'success': True, 'zarr_path': new_zarr_dataset_path, 'dates_combined': [d for d, _ in per_date_items]}


if __name__ == "__main__":
    main()
