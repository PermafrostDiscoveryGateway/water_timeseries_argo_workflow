"""Turn this run's combined breakpoint zarr into the NRT month's PMTiles archive.

``combined_historical_nrt_<date>.zarr`` (written by
``combine_results_into_zarr_historical_archive.py``) carries ``NRTBreakpoint``'s
raw per-lake output for every processed lake, for every month analyzed so far.
That schema matches what ``water_timeseries.utils.pmtiles_build`` expects from a
"full NRT run" (see ``merge_nrt_confidence.TARGET_COLS`` in water-timeseries-v2),
so this script:

1. Opens the most recent combined zarr and pulls out the latest month's slice.
2. Joins that slice to lake polygon geometry (``vector_lake_file``) and writes
   it as a GeoParquet "scored" table -- every lake the run predicted for.
3. Filters the same slice to drained lakes (``water_residual < drain_threshold``)
   and merges them into the running ``nrt_monthly_drain_breaks.parquet``, the
   table ``build_pmtiles_nrt_monthly`` reads to know which lakes drained per month.
4. Calls ``build_pmtiles_nrt_monthly`` in-process to build
   ``nrt_<month>_drainage.pmtiles`` -- no separate CLI/container step needed,
   which also makes this runnable locally against a small test zarr.
"""

import nest_asyncio
# See combine_results_into_zarr_historical_archive.py for why this is needed
# before any zarr store is opened.
nest_asyncio.apply()

import sys
import os
import glob
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.compute as pc
from dotenv import load_dotenv
from loguru import logger

from water_timeseries.utils.pmtiles_build import (
    build_pmtiles_nrt_monthly,
    NRT_MONTHLY_TILE_PROPERTIES,
    NRT_SCORED_TILE_PROPERTIES,
)

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

DEFAULT_DRAIN_THRESHOLD = -0.25


def get_most_recent_combined_zarr(combined_zarr_datasets):
    """Return the most recently created combined_historical_nrt_*.zarr store, or None."""
    zarr_paths = glob.glob(os.path.join(combined_zarr_datasets, "combined_historical_nrt_*.zarr"))
    if not zarr_paths:
        return None
    return max(zarr_paths, key=os.path.getctime)


def collect_geometries(vector_lake_file, wanted_ids, id_column="id_geohash", geometry_column="geometry",
                        batch_size=100_000):
    """Stream {id_geohash: wkb} for wanted_ids out of the (potentially huge) lake table.

    Avoids loading the full multi-GB vector_lake_file into memory just to keep
    a few thousand rows for this month's slice.
    """
    dataset = pa_ds.dataset(vector_lake_file, format="parquet")
    value_set = list(wanted_ids)
    found = {}
    scanner = dataset.scanner(columns=[id_column, geometry_column], batch_size=batch_size)
    for batch in scanner.to_batches():
        ids = batch.column(id_column)
        mask = pc.is_in(ids, value_set=pa.array(value_set, type=ids.type))
        if not pc.any(mask).as_py():
            continue
        matched = batch.filter(mask)
        for gid, wkb in zip(matched.column(id_column).to_pylist(), matched.column(geometry_column).to_pylist()):
            if gid is not None and wkb is not None:
                found[gid] = wkb
    return found


def build_month_dataframe(ds, target_month):
    """Flatten the combined zarr's (id_geohash, date) slice for target_month into a per-lake DataFrame.

    The month-stacking 'date' dimension collides with NRTBreakpoint's own
    per-lake 'date' output column, so combine_results_into_zarr_historical_archive.py
    renames the latter to 'breakpoint_date' before stacking. Undo that here so
    the frame matches the 'date' column name water_timeseries.pmtiles_build expects
    (NRT_SCORED_TILE_PROPERTIES, nrt_scored_rows).
    """
    target_date = pd.Timestamp(f"{target_month}-01")
    if target_date not in pd.to_datetime(ds["date"].values):
        raise ValueError(f"{target_month} not present in combined zarr (available: "
                          f"{sorted({pd.Timestamp(d).strftime('%Y-%m') for d in ds.date.values})})")

    df = ds.sel(date=target_date).to_dataframe().reset_index()
    df = df.drop(columns=["date"])
    if "breakpoint_date" in df.columns:
        df = df.rename(columns={"breakpoint_date": "date"})
    df["id_geohash"] = df["id_geohash"].astype(str)

    # keep_nans=False at analysis time (see process_region_date_new_fast_NRT)
    # means every remaining row already has a real prediction, but guard
    # against any that slipped through with no analysis date.
    if "date" in df.columns:
        df = df[df["date"].notna()].copy()

    return df


def write_scored_geoparquet(df, vector_lake_file, output_path):
    """Join this month's per-lake rows to geometry and write the 'scored' GeoParquet.

    This is what build_pmtiles_nrt_monthly's run_parquet_by_month reads to bake
    the 'scored' layer -- every lake the run predicted for, not just drained ones.
    """
    wanted_ids = set(df["id_geohash"])
    logger.info(f"Collecting geometry for {len(wanted_ids):,} lakes from {vector_lake_file}...")
    geom_by_id = collect_geometries(vector_lake_file, wanted_ids)
    logger.info(f"Found geometry for {len(geom_by_id):,}/{len(wanted_ids):,} lakes")

    df = df.copy()
    df["geometry_wkb"] = df["id_geohash"].map(geom_by_id)
    missing = df["geometry_wkb"].isna().sum()
    if missing:
        logger.warning(f"{missing:,} lakes had no geometry in {vector_lake_file} and will be dropped")
    df = df[df["geometry_wkb"].notna()].copy()

    keep_cols = [c for c in ("id_geohash", *NRT_SCORED_TILE_PROPERTIES) if c in df.columns]
    keep_cols = list(dict.fromkeys(keep_cols))  # de-dupe, keep order
    geometry = gpd.GeoSeries.from_wkb(df.pop("geometry_wkb"), crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(df[keep_cols], geometry=geometry, crs="EPSG:4326")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path)
    logger.info(f"Wrote {len(gdf):,} scored lakes to {output_path}")
    return output_path


def merge_drained_rows(df, target_month, drain_threshold, breaks_file):
    """Append this month's drained lakes into the running nrt_monthly_drain_breaks.parquet.

    Mirrors water_timeseries.scripts.merge_nrt_confidence.merge_nrt_confidence:
    back up the existing table, drop this month's existing rows (safe to
    re-run), concat the new ones in.
    """
    if "water_residual" not in df.columns:
        raise ValueError("Combined zarr is missing 'water_residual' - cannot classify drained lakes")

    drained = df[df["water_residual"] < drain_threshold].copy()
    drained.insert(1, "analysis_month", target_month)
    keep_cols = [c for c in ("id_geohash", "analysis_month", *NRT_MONTHLY_TILE_PROPERTIES) if c in drained.columns]
    keep_cols = list(dict.fromkeys(keep_cols))
    drained = drained[keep_cols]

    logger.info(f"{target_month}: {len(drained):,}/{len(df):,} lakes below drain_threshold={drain_threshold}")

    breaks_file.parent.mkdir(parents=True, exist_ok=True)
    if breaks_file.exists():
        existing = pd.read_parquet(breaks_file)
        backup_path = breaks_file.with_suffix(breaks_file.suffix + ".bak")
        shutil.copy(breaks_file, backup_path)
        logger.info(f"Backed up {breaks_file} -> {backup_path}")
        existing = existing[existing.get("analysis_month") != target_month]
    else:
        existing = pd.DataFrame(columns=drained.columns)

    merged = pd.concat([existing, drained], ignore_index=True)
    merged.to_parquet(breaks_file, index=False)
    logger.info(f"Wrote {len(merged):,} total rows ({len(drained):,} for {target_month}) to {breaks_file}")
    return drained


def main():
    logger.debug("Building NRT PMTiles archive from combined breakpoint zarr")

    env_path = None
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    combined_zarr_datasets = os.environ["combined_zarr_datasets"]
    vector_lake_file = os.environ["vector_lake_file"]
    nrt_precomputed_dir = Path(os.environ["nrt_precomputed_dir"])
    nrt_pmtiles_output_dir = Path(os.environ["nrt_pmtiles_output_dir"])
    drain_threshold = float(os.environ.get("drain_threshold", DEFAULT_DRAIN_THRESHOLD))
    poly_max_zoom = int(os.environ.get("nrt_poly_max_zoom", 14))

    combined_zarr_path = get_most_recent_combined_zarr(combined_zarr_datasets)
    if not combined_zarr_path:
        logger.error(f"No combined_historical_nrt_*.zarr store found under {combined_zarr_datasets}")
        return {'success': False, 'error': 'no combined zarr store found'}

    logger.info(f"Opening {combined_zarr_path}")
    ds = xr.open_zarr(combined_zarr_path)
    ds["id_geohash"] = ds["id_geohash"].astype(str)

    target_month = os.environ.get("nrt_pmtiles_month")
    if not target_month:
        target_month = pd.Timestamp(ds.date.values[-1]).strftime("%Y-%m")
    logger.info(f"Target month: {target_month}")

    df = build_month_dataframe(ds, target_month)
    if df.empty:
        logger.warning(f"No lakes with a prediction for {target_month} - nothing to build")
        return {'success': True, 'target_month': target_month, 'pmtiles': {}}

    scored_path = nrt_precomputed_dir / f"nrt_scored_{target_month}.parquet"
    write_scored_geoparquet(df, vector_lake_file, scored_path)

    breaks_file = nrt_precomputed_dir / "nrt_monthly_drain_breaks.parquet"
    drained = merge_drained_rows(df, target_month, drain_threshold, breaks_file)

    if drained.empty:
        logger.warning(f"No drained lakes for {target_month} - skipping pmtiles build "
                        f"(the base archive already covers every stable lake)")
        return {'success': True, 'target_month': target_month, 'scored_path': str(scored_path), 'pmtiles': {}}

    outputs = build_pmtiles_nrt_monthly(
        breaks_parquet=breaks_file,
        geometry_parquet=vector_lake_file,
        output_dir=nrt_pmtiles_output_dir,
        months=[target_month],
        run_parquet_by_month={target_month: scored_path},
        poly_max_zoom=poly_max_zoom,
    )

    for month, path in outputs.items():
        logger.success(f"[{month}] wrote {path}")

    return {
        'success': True,
        'target_month': target_month,
        'scored_path': str(scored_path),
        'pmtiles': {month: str(path) for month, path in outputs.items()},
    }


if __name__ == "__main__":
    main()
