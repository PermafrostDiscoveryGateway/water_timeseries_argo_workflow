"""
Final, once-a-month retry for lake IDs that download_region_missing_ids.py
has confirmed have no Dynamic World data.

Dynamic World ingestion for a just-finished month can lag by a couple of
weeks. download_region_missing_ids.py runs daily starting a few days into
the month and, when it gets a confirmed-empty result for a batch of lakes,
parks those IDs in a no_data_ids_<region>_<date>.json file instead of
retrying them every day (see mark_no_data() there) - that would just waste
Earth Engine requests re-confirming the same "not ingested yet" answer.

This script is that deferred retry: gated to only run once we're well into
the month (see utils.date_gate.should_run_last_attempts), it re-requests
exactly the parked IDs, one more time each. Whichever come back with data
are recovered like any other backfill; whichever are still empty are
marked `final` in the same JSON file, so:
  - this script itself won't request them again on a later run this month
  - other tooling (e.g. a completion checker) can treat them as permanently
    unavailable rather than counting them against completion percentage

Intended to be run as its own cron job, one per region, after the daily
download_region_missing_ids.py job has had the whole first half of the
month to work through genuinely-retryable failures.
"""
import ee
import gc
import geemap
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
from utils.date_gate import should_run_last_attempts, most_recent_summer_month
from download_region_missing_ids import (
    normalize_id,
    normalize_id_set,
    no_data_ids_path,
    load_no_data_ids,
    save_no_data_ids,
    mark_no_data,
    download_id_batches,
)

# Downloads fail transiently often enough that one pass isn't a fair test of
# "is there really no data" - retry parked IDs this many times (each pass
# only re-targeting whatever the previous pass didn't recover) before
# marking anything still missing as permanently no-data.
LAST_ATTEMPT_RETRIES = 2


def write_recovered_flag(recovered: bool):
    """Write true/false for Argo to pick up as an output parameter.

    Lets the Argo template that runs this script skip the follow-up merge
    step on every day it had nothing to do (i.e. every day before day 15,
    and any day after with nothing parked), instead of re-merging daily.
    """
    output_dir = Path(os.environ.get('LAST_ATTEMPT_OUTPUT_DIR', '/tmp/last_attempt_status'))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'recovered.txt').write_text('true' if recovered else 'false')


def main():
    write_recovered_flag(False)
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

    if not should_run_last_attempts():
        logger.debug("Not yet day 15 of the month (or last month wasn't in season) - exiting")
        return 0

    target_month = most_recent_summer_month()
    date_to_run = target_month.strftime("%Y-%m")
    logger.info(f"Running last-attempt check for {REGION} / {date_to_run}")

    current_download_dir = dynamic_world_download_dir / REGION / f'download_{date_to_run}'
    current_download_dir.mkdir(parents=True, exist_ok=True)

    # ========== LOAD PARKED NO-DATA IDs ==========
    no_data_path = no_data_ids_path(dynamic_world_data_dir, REGION, date_to_run)
    known_no_data = load_no_data_ids(no_data_path)
    pending_ids = {id_ for id_, info in known_no_data.items() if not info.get('final')}

    if not pending_ids:
        logger.info(f"No pending no-data IDs parked for {REGION} / {date_to_run} - nothing to do")
        return 0

    logger.info(f"Found {len(pending_ids):,} lakes parked as no-data, giving them one final check")

    # Guard against anything that made it into the merged file since being
    # parked (e.g. a manual re-run of missing_ids before this ran).
    merged_file_path = os.path.join(dynamic_world_data_dir, 'merge', f"dw_{REGION}_{date_to_run}.nc")
    if Path(merged_file_path).exists():
        ds_merged = xr.open_dataset(merged_file_path)
        ids_in_merged_file = normalize_id_set(ds_merged['id_geohash'].values.tolist())
        ds_merged.close()
        pending_ids -= ids_in_merged_file

    if not pending_ids:
        logger.info(f"✅ All previously-parked IDs for {REGION} / {date_to_run} are already in the merged file")
        return 0

    region_lake_file = Path(region_lake_polygons_dir) / f"{REGION}_lake_polygons.parquet"
    logger.info(f"Loading pre-split lake vector file for {REGION}: {region_lake_file}")
    gdf_region = gpd.read_parquet(region_lake_file)
    gdf_region['id_geohash'] = gdf_region['id_geohash'].apply(normalize_id)

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

    # ========== FINAL ATTEMPT(S) ==========
    # Downloads fail transiently often enough that a single pass isn't a fair
    # test of "is there really no data" - so give it a couple of tries, each
    # one only re-targeting whatever the previous pass didn't recover.
    recovered_ids = set()
    remaining_ids = set(pending_ids)
    had_real_error = False

    for attempt in range(1, LAST_ATTEMPT_RETRIES + 1):
        if not remaining_ids:
            break
        logger.info(
            f"\n{'#' * 80}\nLast-attempt pass {attempt}/{LAST_ATTEMPT_RETRIES}: "
            f"{len(remaining_ids):,} lakes\n{'#' * 80}"
        )
        gdf_pending = gdf_region[gdf_region['id_geohash'].isin(remaining_ids)]
        run_label = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        result = download_id_batches(
            downloader=downloader,
            gdf_ids=gdf_pending,
            id_list=sorted(remaining_ids),
            date_to_run=date_to_run,
            current_download_dir=current_download_dir,
            run_label=f"{run_label}_pass{attempt}",
            file_prefix='last_attempt',
        )
        recovered_ids.update(result['recovered_ids'])
        remaining_ids -= result['recovered_ids']
        had_real_error = had_real_error or result['had_real_error']

    # Anything still unrecovered after every pass - confirmed no-data or
    # repeatedly below the completion threshold - is treated as final here:
    # this is the last attempt, there is no future run left to retry it against.
    still_no_data_ids = remaining_ids

    for id_ in recovered_ids:
        known_no_data.pop(id_, None)
    if still_no_data_ids:
        mark_no_data(known_no_data, still_no_data_ids, final=True)
    save_no_data_ids(no_data_path, known_no_data)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"LAST-ATTEMPT SUMMARY for {REGION} / {date_to_run}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Parked no-data IDs checked: {len(pending_ids):,}")
    logger.info(f"Recovered (ingestion caught up): {len(recovered_ids):,}")
    logger.info(f"Confirmed permanently no-data: {len(still_no_data_ids):,}")

    if recovered_ids:
        write_recovered_flag(True)

    if had_real_error:
        logger.error(
            f"❌ At least one batch hit a real (non-no-data) error during the final attempts for "
            f"{REGION} / {date_to_run} - investigate before treating this region as done"
        )
        return 1

    logger.info(f"✅ Last-attempt check complete for {REGION} / {date_to_run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
