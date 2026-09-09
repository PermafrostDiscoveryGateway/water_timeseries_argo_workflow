"""
Read-only completion auditor for the near-real-time pipeline.

Runs after a merge pass and reports, per region, what fraction of the
region's expected lake IDs are present in that region's merged file
(dw_<region>_<date>.nc). Used by the Argo pipeline to decide whether the
merge -> backfill loop needs another round before moving on to processing.

Does not modify any data and always exits 0 - it's a reporter, not a gate.
Threshold matches ACCEPTABLE_MERGE_THRESHOLD in merge_recent_downloads.py.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import xarray as xr
from dotenv import load_dotenv
from loguru import logger

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.region_boundaries import get_region_boundaries, get_small_regions
from utils.date_gate import is_test_run, most_recent_summer_month
from near_real_time.download_region_missing_ids import (
    get_region_lakes,
    get_historical_valid_ids,
    normalize_id,
    normalize_id_set,
    no_data_ids_path,
    load_no_data_ids,
)

COMPLETION_THRESHOLD = 0.98


def get_confirmed_no_data_ids(dynamic_world_data_dir, region, date_to_run):
    """Lake IDs download_region_last_attempts.py has permanently confirmed have no
    Dynamic World data for this region/date - these can never be "present" in the
    merged file, so they shouldn't count against completion once confirmed final.
    Provisional (not-yet-final) no-data IDs still count as missing here - they're
    just deferred to the last-attempts check, not yet known to be unrecoverable.
    """
    no_data_record = load_no_data_ids(no_data_ids_path(dynamic_world_data_dir, region, date_to_run))
    return {id_ for id_, info in no_data_record.items() if info.get('final')}


def region_completion(region, region_boundaries, gdf, historical_valid_ids, merged_file,
                       dynamic_world_data_dir, date_to_run):
    gdf_region = get_region_lakes(gdf, region_boundaries, region)
    region_ids = set(gdf_region['id_geohash'].values.tolist())
    if historical_valid_ids is not None:
        region_ids = region_ids & historical_valid_ids

    confirmed_no_data_ids = get_confirmed_no_data_ids(dynamic_world_data_dir, region, date_to_run)
    obtainable_ids = region_ids - confirmed_no_data_ids

    if not obtainable_ids:
        return 1.0, 0, 0, len(confirmed_no_data_ids)

    if not Path(merged_file).exists():
        return 0.0, 0, len(obtainable_ids), len(confirmed_no_data_ids)

    ds = xr.open_dataset(merged_file)
    try:
        ids_in_file = normalize_id_set(ds['id_geohash'].values.tolist())
    finally:
        ds.close()

    present = len(ids_in_file & obtainable_ids)
    return present / len(obtainable_ids), present, len(obtainable_ids), len(confirmed_no_data_ids)


def main():
    env_path = sys.argv[1] if len(sys.argv) > 1 else None
    if env_path:
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    dynamic_world_data_dir = os.environ['dynamic_world_data']
    vector_lake_file = os.environ['vector_lake_file']

    output_dir = Path(os.environ.get('COMPLETION_OUTPUT_DIR', '/tmp'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Same "should we even be running for this month" gate the download/merge
    # scripts use, so the checker doesn't chase a date nobody is processing.
    summer_months = [6, 7, 8, 9]
    TODAY = datetime.now()
    if is_test_run():
        should_run = True
        target_month = most_recent_summer_month(TODAY)
        logger.debug(f"test_run=True - bypassing day-of-month/season gate, using {target_month.strftime('%Y-%m')}")
    else:
        should_run = (TODAY.month - 1) in summer_months and TODAY.day > 3
        target_month = datetime(TODAY.year, TODAY.month - 1, 1)

    if not should_run:
        logger.info("Outside the operating window this month - reporting all-complete so the pipeline proceeds")
        (output_dir / 'all_complete.txt').write_text('true')
        (output_dir / 'incomplete_regions.json').write_text('[]')
        return 0

    date_to_run = target_month.strftime("%Y-%m")

    region_boundaries = get_region_boundaries()
    all_regions = list(region_boundaries.keys())
    test_run = os.environ.get("test_run")
    if test_run and test_run.lower() == 'true':
        all_regions = list(get_small_regions().keys())

    logger.info("Loading lake vector file...")
    gdf = gpd.read_parquet(vector_lake_file)
    gdf['id_geohash'] = gdf['id_geohash'].apply(normalize_id)

    historical_valid_ids = get_historical_valid_ids(dynamic_world_data_dir)

    region_status = {}
    incomplete_regions = []
    for region in all_regions:
        merged_file = os.path.join(dynamic_world_data_dir, 'merge', f"dw_{region}_{date_to_run}.nc")
        pct, present, total, confirmed_no_data = region_completion(
            region, region_boundaries, gdf, historical_valid_ids, merged_file,
            dynamic_world_data_dir, date_to_run
        )
        region_status[region] = {
            'completion_pct': pct,
            'ids_present': present,
            'ids_expected': total,
            'ids_confirmed_no_data': confirmed_no_data,
        }

        status_icon = "✅" if pct >= COMPLETION_THRESHOLD else "⚠️"
        logger.info(
            f"{status_icon} {region}: {pct:.2%} complete ({present:,}/{total:,}, "
            f"{confirmed_no_data:,} confirmed no-data excluded) for {date_to_run}"
        )

        if pct < COMPLETION_THRESHOLD:
            incomplete_regions.append(region)

    all_complete = len(incomplete_regions) == 0

    (output_dir / 'all_complete.txt').write_text('true' if all_complete else 'false')
    (output_dir / 'incomplete_regions.json').write_text(json.dumps(incomplete_regions))

    checkpoint = {
        'date_to_run': date_to_run,
        'checked_at': datetime.utcnow().isoformat() + 'Z',
        'threshold': COMPLETION_THRESHOLD,
        'all_complete': all_complete,
        'incomplete_regions': incomplete_regions,
        'regions': region_status,
    }
    checkpoint_dir = Path(dynamic_world_data_dir).parent / 'checkpoint'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"completion_{date_to_run}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"Wrote checkpoint: {checkpoint_file}")

    if all_complete:
        logger.info(f"✅ All regions >= {COMPLETION_THRESHOLD:.0%} complete for {date_to_run}")
    else:
        logger.warning(f"⚠️ Regions still below {COMPLETION_THRESHOLD:.0%}: {incomplete_regions}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
