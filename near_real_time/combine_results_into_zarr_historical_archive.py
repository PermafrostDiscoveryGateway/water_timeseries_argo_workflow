import nest_asyncio
# Zarr v3's async I/O keeps a per-thread event loop; running under a debugger
# (e.g. PyCharm's pydevd) can hand a later store-open a Future tied to a stale
# loop ("Task ... got Future ... attached to a different loop"). Patching
# asyncio to tolerate re-entrant loops here avoids that crash.
nest_asyncio.apply()

from dotenv import load_dotenv
import os
import glob
import time
import numpy as np
import pandas as pd
from datetime import datetime
import sys
from loguru import logger
from pathlib import Path
import xarray as xr
from utils.region_boundaries import get_region_boundaries
from utils.date_gate import is_test_run, most_recent_summer_month
# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# A breakpoint zarr store that was written to within this window is assumed to
# still be mid-write, so it's excluded from the merge rather than folded in as a partial.
STALE_THRESHOLD_SECONDS = 60 * 2


def get_most_recent_combined_zarr(combined_zarr_datasets):
    """Return the most recently created combined_historical_nrt_*.zarr store, or None if there isn't one."""
    zarr_paths = glob.glob(os.path.join(combined_zarr_datasets, "combined_historical_nrt_*.zarr"))
    if not zarr_paths:
        return None
    return max(zarr_paths, key=os.path.getctime)


def zarr_store_age_seconds(zarr_path):
    """Seconds since the most recently modified file inside a zarr store."""
    files = glob.glob(os.path.join(zarr_path, "**", "*"), recursive=True)
    last_modified = max((os.path.getmtime(f) for f in files), default=os.path.getmtime(zarr_path))
    return time.time() - last_modified


def discover_available_dates(output_dir, regions):
    """Return every date (e.g. '2026-07') that has at least one region breakpoint zarr in output_dir."""
    dates = set()
    dates_by_region = {}
    for region in regions:
        breakpoint_dir = os.path.join(output_dir, region, 'breakpoint_zarr')
        suffix = ".zarr"
        region_dates = []
        for zarr_path in glob.glob(os.path.join(breakpoint_dir, f"breakpoints_*{suffix}")):
            filename = os.path.basename(zarr_path)
            date = filename[len("breakpoints_"):-len(suffix)]
            dates.add(date)
            region_dates.append(date)
        dates_by_region[region] = sorted(region_dates)
        if region_dates:
            logger.debug(f"{region}: found breakpoint zarr for {sorted(region_dates)}")
        else:
            logger.warning(f"{region}: no breakpoint zarr files found under {breakpoint_dir}")
    return sorted(dates)


def merge_regions_for_date(date, output_dir, regions):
    """Open and merge every region's breakpoint zarr for a single date along id_geohash, skipping missing/stale stores."""
    results_paths_to_merge = []
    missing_regions = []
    stale_regions = []
    for region in regions:
        path = os.path.join(output_dir, region, 'breakpoint_zarr', f"breakpoints_{date}.zarr")
        if not os.path.exists(path):
            missing_regions.append(region)
            logger.warning(f"[{date}] Breakpoint zarr does not exist, skipping: {path}")
            continue
        age_seconds = zarr_store_age_seconds(path)
        if age_seconds < STALE_THRESHOLD_SECONDS:
            stale_regions.append(region)
            logger.warning(
                f"[{date}] Breakpoint zarr was modified {age_seconds:.0f}s ago (< {STALE_THRESHOLD_SECONDS}s) "
                f"and may still be writing, skipping: {path}"
            )
            continue
        results_paths_to_merge.append(path)

    logger.info(
        f"[{date}] Region availability: {len(results_paths_to_merge)}/{len(regions)} usable"
        f"{f', missing: {missing_regions}' if missing_regions else ''}"
        f"{f', stale/still-writing: {stale_regions}' if stale_regions else ''}"
    )

    if not results_paths_to_merge:
        logger.error(
            f"[{date}] No complete breakpoint zarr datasets available to merge - this date will be "
            f"dropped from this run. Missing for every region: {missing_regions or 'n/a'}; "
            f"stale/still-writing: {stale_regions or 'n/a'}"
        )
        return None

    if missing_regions or stale_regions:
        logger.warning(
            f"[{date}] Merging PARTIAL data - {len(missing_regions)} region(s) missing, "
            f"{len(stale_regions)} region(s) stale/still-writing. This month's combined archive "
            f"will be incomplete for: {missing_regions + stale_regions}"
        )

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
    logger.debug(f"Combining regional breakpoint zarr datasets into a single archive")
    env_path = None
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    combined_zarr_datasets = os.environ.get("combined_zarr_datasets")

    existing_combined_path = get_most_recent_combined_zarr(combined_zarr_datasets)
    if existing_combined_path:
        logger.info(f"Found existing combined archive: {existing_combined_path}")
    else:
        logger.info(f"No existing combined archive found in {combined_zarr_datasets} - starting a new one")

    MANUAL = bool(os.environ.get("MANUAL"))

    SHOULD_RUN = False
    summer_months = [6, 7, 8, 9]
    TODAY = datetime.now()
    TODAY_MONTH = TODAY.month
    TODAY_YEAR = TODAY.year
    # Previous calendar month, wrapping the year when today is in January.
    target_date = (pd.Timestamp(year=TODAY_YEAR, month=TODAY_MONTH, day=1) - pd.DateOffset(months=1)).strftime("%Y-%m")

    if MANUAL:
        SHOULD_RUN = True
        logger.debug(f"MANUAL=True - bypassing day-of-month/season gate. Should run: {SHOULD_RUN}")
        logger.debug(f"Target date: {target_date}")
    elif is_test_run():
        SHOULD_RUN = True
        target_date = most_recent_summer_month(TODAY).strftime("%Y-%m")
        logger.debug(f"test_run=True - bypassing day-of-month/season gate, using {target_date}")
    elif (TODAY_MONTH - 1) in summer_months:
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 3:
            SHOULD_RUN = True
            logger.debug(f"TODAY_DAY: {TODAY_DAY} - Should run: {SHOULD_RUN}")
            logger.debug(f"Target date: {target_date}")

    if not SHOULD_RUN:
        logger.info("Skipping processing - conditions not met")
        return

    output_dir = os.environ['output_dir']

    regions = list(get_region_boundaries().keys())

    ds_existing = None
    existing_dates = set()
    if existing_combined_path:
        ds_existing = xr.open_zarr(existing_combined_path)
        ds_existing['id_geohash'] = ds_existing['id_geohash'].astype(str)
        existing_dates = {pd.Timestamp(d).strftime("%Y-%m") for d in ds_existing.date.values}

    # Cheap up-front check: if every available regional month (including any
    # earlier ones a prior run may have skipped) is already in the combined
    # archive, there's nothing to do - skip before doing any of the
    # (expensive) per-region merging below.
    available_dates = discover_available_dates(output_dir, regions)
    if available_dates and set(available_dates) <= existing_dates:
        logger.info(
            f"Combined archive already contains every available regional month "
            f"(through {max(available_dates)}) - already run, skipping"
        )
        return

    # Combine the current target month plus any earlier available months that
    # aren't in the existing archive yet (e.g. a prior run that skipped/failed
    # a month), rather than just the single most recent one.
    if not available_dates:
        logger.error(f"No breakpoint zarr datasets found under {output_dir} - aborting")
        return

    target_dates = sorted(set(available_dates) | {target_date})
    target_dates = [date for date in target_dates if date not in existing_dates]
    if not target_dates:
        logger.info("All available dates are already present in the existing combined archive - nothing to do")
        return
    logger.debug(f"Found {len(target_dates)} date(s) not yet in the combined archive: {target_dates}")

    per_date_items = [
        (date, ds) for date, ds in (
            (date, merge_regions_for_date(date, output_dir, regions)) for date in target_dates
        )
        if ds is not None
    ]

    merged_dates = {date for date, _ in per_date_items}
    dropped_dates = [date for date in target_dates if date not in merged_dates]
    if dropped_dates:
        logger.warning(
            f"{len(dropped_dates)} target date(s) will NOT be added to the combined archive this run "
            f"because no region had usable breakpoint data for them: {dropped_dates}. "
            f"See the '[<date>] Region availability' / 'this date will be dropped' log lines above "
            f"for the per-region breakdown."
        )

    if not per_date_items:
        logger.error("No complete breakpoint zarr datasets available to merge - aborting")
        return

    target_dates = [date for date, _ in per_date_items]

    # Some per-region stores have a per-lake 'date' variable (breakpoint date),
    # which would collide with the 'date' dimension we're about to create to
    # stack months - rename it out of the way when present. Current breakpoint
    # output instead uses date_before_break/date_after_break/date_break, which
    # don't collide and are left as-is; those are enough on their own.
    per_date_datasets = [
        (ds.rename({"date": "breakpoint_date"}) if "date" in ds.variables or "date" in ds.dims else ds)
        .assign_coords(date=pd.Timestamp(f"{date}-01"))
        .expand_dims("date")
        for date, ds in per_date_items
    ]

    if len(per_date_datasets) > 1:
        ds_new_merged = xr.concat(per_date_datasets, dim="date")
    else:
        ds_new_merged = per_date_datasets[0]

    if ds_existing is not None:
        logger.info(f"Appending {len(target_dates)} new month(s) to existing combined archive...")
        ds_combined = xr.concat([ds_existing, ds_new_merged], dim="date", join="outer")
        latest_date = pd.Timestamp(ds_combined.date.values[-1]).strftime("%Y-%m")
    else:
        ds_combined = ds_new_merged
        latest_date = target_dates[-1]

    # Concatenation/alignment can leave ragged dask chunks (especially with join="outer"
    # padding mismatched id_geohash coordinates), which zarr refuses to write. Consolidate
    # to a single chunk per dimension before saving.
    ds_combined = ds_combined.chunk({"id_geohash": -1, "date": -1})

    new_zarr_dataset_name = f"combined_historical_nrt_{latest_date}.zarr"
    new_zarr_dataset_path = os.path.join(combined_zarr_datasets, new_zarr_dataset_name)

    logger.info(f"Saving combined result to {new_zarr_dataset_path}")
    ds_combined.to_zarr(new_zarr_dataset_path, mode='w', align_chunks=True)

    logger.success(f"Combined zarr dataset written to {new_zarr_dataset_path}")


if __name__ == "__main__":
    main()

