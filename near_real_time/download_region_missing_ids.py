"""
Targeted backfill for lake IDs that are missing from a region's merged file.

Unlike download_region.py (which re-downloads entire 1x1 degree grid tiles),
this script:
  1. Computes exactly which lake IDs are missing from the region's merged
     NetCDF file (bounding-box lakes, restricted to the historical baseline,
     minus whatever IDs are already in the merged file).
  2. Requests only those specific lakes from Earth Engine, in batches.
  3. Writes results into the same downloads/<region>/download_<date>/
     directory as the regular per-tile downloads, using a filename that
     matches the DW_<date>_*.nc pattern merge_new_results() already globs
     for. The next scheduled merge run folds these in automatically via its
     existing combine + dedup logic - no merge-side changes needed.

Intended to be run as its own cron job, one per region, after a normal merge
run has reported missing IDs for that region. Safe to re-run: IDs already
captured by a previous run of this script (but not yet merged) are skipped,
and batches that come back too incomplete are discarded so they're retried
on the next run instead of being silently accepted.
"""
import ee
import gc
import geemap
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import xarray as xr
from dotenv import load_dotenv
from loguru import logger
from water_timeseries.downloader import EarthEngineDownloader

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.region_boundaries import get_region_boundaries
from utils.date_gate import is_test_run, most_recent_summer_month

# Accept a batch as complete if this fraction of its requested lakes come
# back. Matches TILE_COMPLETION_THRESHOLD in utils/helper_functions.py and
# COMPLETION_THRESHOLD/ACCEPTABLE_MERGE_THRESHOLD in merge_recent_downloads.py.
COMPLETION_THRESHOLD = 0.98

# How many lakes to request per Earth Engine call.
BATCH_SIZE = 500


def normalize_id(value):
    """Make id_geohash values comparable across netCDF (numpy/bytes) and geopandas (str) sources."""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def normalize_id_set(values):
    return {normalize_id(v) for v in values}


def get_region_lakes(gdf, region_boundaries, region):
    """Filter the full lake vector file down to the ones inside a region's bounding box."""
    bounds = region_boundaries[region]
    x_min_start = bounds['X_MIN_START']
    x_min_end = bounds['X_MIN_END']
    y_min_start = bounds['Y_MIN_START']
    y_min_end = bounds['Y_MIN_END']

    geom_type = gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else None
    if geom_type in ['Polygon', 'MultiPolygon']:
        centroids = gdf.geometry.centroid
        x_coords = centroids.x
        y_coords = centroids.y
    elif geom_type == 'Point':
        x_coords = gdf.geometry.x
        y_coords = gdf.geometry.y
    else:
        rep_points = gdf.geometry.representative_point()
        x_coords = rep_points.x
        y_coords = rep_points.y

    mask = (x_coords >= x_min_start) & (x_coords <= x_min_end) & \
           (y_coords >= y_min_start) & (y_coords <= y_min_end)
    return gdf[mask]


def get_historical_valid_ids(dynamic_world_data_dir):
    """Same historical-baseline lookup used by the download step and merge-side verification."""
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
    if not all_dynamic_world_files:
        return None
    historical_file = max(all_dynamic_world_files, key=os.path.getctime)
    ds_historical = xr.open_dataset(historical_file)
    valid_ids = normalize_id_set(ds_historical['id_geohash'].values.tolist())
    ds_historical.close()
    return valid_ids


def get_already_backfilled_ids(download_dir, date_to_run):
    """IDs already captured by a previous run of this script but not yet folded in by a merge."""
    already_have = set()
    pattern = str(download_dir / f'DW_{date_to_run}_missing_backfill_*.nc')
    for f in glob.glob(pattern):
        try:
            ds = xr.open_dataset(f)
            already_have.update(normalize_id_set(ds['id_geohash'].values.tolist()))
            ds.close()
        except Exception as e:
            logger.warning(f"Could not read existing backfill file {f}: {e}")
    return already_have


def main():
    env_path = None
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    REGION = os.environ.get("region_name", "TEST")
    logger.info(f"=== REGION FROM ENV: '{REGION}' ===")

    region_boundaries = get_region_boundaries()
    if REGION not in region_boundaries:
        logger.error(f"Region '{REGION}' not found in boundaries! Available: {list(region_boundaries.keys())}")
        return 1

    dynamic_world_data_dir = os.environ['dynamic_world_data']
    dynamic_world_download_dir = Path(os.environ['dynamic_world_downloads'])
    region_lake_polygons_dir = os.environ['region_lake_polygons_dir']
    project = os.environ['project']

    # ========== DETERMINE DATE TO RUN (same window as the regular download jobs) ==========
    SHOULD_RUN = False
    summer_months = [6, 7, 8, 9]
    TODAY = datetime.now()
    TODAY_MONTH = TODAY.month
    target_month = None
    if is_test_run():
        target_month = most_recent_summer_month(TODAY)
        SHOULD_RUN = True
        logger.debug(f"test_run=True - bypassing day-of-month/season gate, using {target_month.strftime('%Y-%m')}")
    elif TODAY_MONTH - 1 in summer_months:
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 2:
            SHOULD_RUN = True
            target_month = datetime(TODAY.year, TODAY_MONTH - 1, 1)

    if not SHOULD_RUN:
        logger.debug("Too early in the month to run - exiting")
        return 0

    date_to_run = target_month.strftime("%Y-%m")
    logger.info(f"Backfilling missing IDs for {REGION} / {date_to_run}")

    # ========== FIGURE OUT WHICH IDs ARE ACTUALLY MISSING ==========
    merged_file_path = os.path.join(dynamic_world_data_dir, 'merge', f"dw_{REGION}_{date_to_run}.nc")
    if not Path(merged_file_path).exists():
        logger.error(f"No merged file found at {merged_file_path} - run the regular merge job first")
        return 1

    ds_merged = xr.open_dataset(merged_file_path)
    ids_in_merged_file = normalize_id_set(ds_merged['id_geohash'].values.tolist())
    ds_merged.close()
    logger.info(f"Merged file currently has {len(ids_in_merged_file):,} IDs")

    region_lake_file = Path(region_lake_polygons_dir) / f"{REGION}_lake_polygons.parquet"
    logger.info(f"Loading pre-split lake vector file for {REGION}: {region_lake_file}")
    gdf_region = gpd.read_parquet(region_lake_file)
    gdf_region['id_geohash'] = gdf_region['id_geohash'].apply(normalize_id)

    region_ids = set(gdf_region['id_geohash'].values.tolist())
    logger.info(f"Region {REGION} has {len(region_ids):,} lakes")

    historical_valid_ids = get_historical_valid_ids(dynamic_world_data_dir)
    if historical_valid_ids is not None:
        before = len(region_ids)
        region_ids = region_ids & historical_valid_ids
        logger.info(f"Restricted to historical baseline: {before:,} -> {len(region_ids):,} eligible lakes")
    else:
        logger.warning("Could not load historical valid IDs - using full bounding-box lake set")

    current_download_dir = dynamic_world_download_dir / REGION / f'download_{date_to_run}'
    current_download_dir.mkdir(parents=True, exist_ok=True)

    already_backfilled_ids = get_already_backfilled_ids(current_download_dir, date_to_run)
    if already_backfilled_ids:
        logger.info(f"Found {len(already_backfilled_ids):,} IDs already captured by a previous backfill run")

    missing_ids = region_ids - ids_in_merged_file - already_backfilled_ids

    if not missing_ids:
        logger.info(f"✅ No missing IDs left for {REGION} / {date_to_run} - nothing to backfill")
        return 0

    logger.info(f"Need to backfill {len(missing_ids):,} missing lake IDs for {REGION} / {date_to_run}")

    gdf_missing = gdf_region[gdf_region['id_geohash'].isin(missing_ids)]

    # ========== INITIALIZE EARTH ENGINE ==========
    os.environ["EE_PROJECT"] = project
    try:
        ee.Initialize(project=project)
        logger.debug("Earth engine successfully initialized")
    except Exception as e:
        logger.debug(f"Failed to initialize earth engine: {e}")
    try:
        geemap.ee_initialize(project=project)
        logger.debug("Initialized geemap")
    except Exception as e:
        logger.debug(f"Failed to initialize geemap: {e}")

    if not hasattr(geemap, 'ee_initialize'):
        logger.warning("geemap.ee_initialize missing, adding runtime patch")

        def ee_initialize(project=None, **kwargs):
            if project:
                ee.Initialize(project=project, **kwargs)
            else:
                ee.Initialize(**kwargs)

        geemap.ee_initialize = ee_initialize
        logger.info("Runtime patch applied to geemap")

    downloader = EarthEngineDownloader(ee_project=project)

    # ========== DOWNLOAD IN BATCHES ==========
    missing_id_list = sorted(missing_ids)
    batches = [missing_id_list[i:i + BATCH_SIZE] for i in range(0, len(missing_id_list), BATCH_SIZE)]
    logger.info(f"Requesting {len(missing_id_list):,} lakes in {len(batches)} batch(es) of up to {BATCH_SIZE}")

    run_label = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    total_recovered = 0
    total_discarded = 0

    for batch_idx, batch_ids in enumerate(batches):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Batch {batch_idx + 1}/{len(batches)}: {len(batch_ids)} lakes")
        logger.info(f"{'=' * 80}")

        batch_id_set = set(batch_ids)
        gdf_batch = gdf_missing[gdf_missing['id_geohash'].isin(batch_id_set)]
        outfile = current_download_dir / f'DW_{date_to_run}_missing_backfill_{run_label}_{batch_idx}.nc'

        n_features = len(gdf_batch)
        max_total_requests = min(100, n_features) if n_features > 500 else 500

        try:
            ds_dl = downloader.download_dw_monthly(
                gdf=gdf_batch,
                max_total_requests=max_total_requests,
                n_parallel=2,
                date_list=[date_to_run],
                save_to_file=outfile
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx} failed: {e}")
            continue

        if ds_dl is None:
            logger.warning(f"Batch {batch_idx}: no data returned")
            continue

        downloaded_ids = normalize_id_set(ds_dl['id_geohash'].values.tolist())
        completion_pct = len(downloaded_ids) / len(batch_id_set) if batch_id_set else 1.0

        ds_dl.close()
        del ds_dl
        gc.collect()

        if completion_pct < COMPLETION_THRESHOLD:
            logger.warning(
                f"Batch {batch_idx}: only {len(downloaded_ids)}/{len(batch_id_set)} lakes came back "
                f"({completion_pct:.2%}, below {COMPLETION_THRESHOLD:.0%} threshold) - discarding, will retry next run")
            try:
                outfile.unlink()
            except Exception as e:
                logger.warning(f"Could not remove incomplete batch file {outfile}: {e}")
            total_discarded += len(batch_id_set)
            continue

        logger.info(
            f"✅ Batch {batch_idx}: {len(downloaded_ids)}/{len(batch_id_set)} lakes "
            f"({completion_pct:.2%}) - keeping")
        total_recovered += len(downloaded_ids)

    still_missing = len(missing_ids) - total_recovered

    logger.info(f"\n{'=' * 80}")
    logger.info(f"BACKFILL SUMMARY for {REGION} / {date_to_run}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Missing before this run: {len(missing_ids):,}")
    logger.info(f"Recovered this run: {total_recovered:,}")
    logger.info(f"Discarded (below threshold, will retry next run): {total_discarded:,}")
    logger.info(f"Still missing: {still_missing:,}")
    logger.info("Run the regular merge job again to fold these into the merged file.")

    if still_missing > 0:
        logger.error(f"❌ {still_missing:,} lakes still missing for {REGION} / {date_to_run} - run again to retry")
        return 1

    logger.info(f"✅ All missing lakes recovered for {REGION} / {date_to_run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
