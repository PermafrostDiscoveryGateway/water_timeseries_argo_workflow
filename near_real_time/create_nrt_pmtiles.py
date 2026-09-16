"""Turn this run's combined breakpoint zarr into the NRT month's PMTiles archive.

``combined_historical_nrt_<date>.zarr`` (written by
``combine_results_into_zarr_historical_archive.py``) carries ``NRTBreakpoint``'s
raw per-lake output for every processed lake, for every month analyzed so far.
This script:

1. Opens the most recent combined zarr and pulls out the latest month's slice.
2. Filters that slice to drained lakes (``water_residual < drain_threshold``)
   and merges them into the running ``nrt_monthly_drain_breaks.parquet``, the
   table ``build_pmtiles_nrt_monthly`` reads to know which lakes drained per month.
3. Calls ``build_pmtiles_nrt_monthly`` in-process to build
   ``nrt_<month>_drainage.pmtiles`` -- no separate CLI/container step needed,
   which also makes this runnable locally against a small test zarr.

Note: the ``water_timeseries.utils.pmtiles_build.build_pmtiles_nrt_monthly``
on this branch (``ncsa-water-timeseries``) only builds the ``drained``/
``drained_points`` overlay layer -- it joins geometry from ``geometry_parquet``
itself, so no separate geometry join is needed here. A newer ``scored`` layer
(every non-drained lake also carrying its NRT prediction) exists on the
``fix/drainage-confidence`` branch but is not merged in here yet; a
non-drained lake's tooltip falls back to the shared base archive until it is.
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

import pandas as pd
import xarray as xr
from dotenv import load_dotenv
from loguru import logger

from water_timeseries.utils.pmtiles_build import (
    build_pmtiles_nrt_monthly,
    NRT_MONTHLY_TILE_PROPERTIES,
)
from water_timeseries.utils.nrt_postprocessing import DRAIN_THRESHOLD, drained_mask

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def get_most_recent_combined_zarr(combined_zarr_datasets):
    """Return the most recently created combined_historical_nrt_*.zarr store, or None."""
    zarr_paths = glob.glob(os.path.join(combined_zarr_datasets, "combined_historical_nrt_*.zarr"))
    if not zarr_paths:
        return None
    return max(zarr_paths, key=os.path.getctime)


def build_month_dataframe(ds, target_month):
    """Flatten the combined zarr's (id_geohash, date) slice for target_month into a per-lake DataFrame.

    The month-stacking 'date' dimension collides with NRTBreakpoint's own
    per-lake 'date' output column, so combine_results_into_zarr_historical_archive.py
    renames the latter to 'breakpoint_date' before stacking. Undo that here so
    a 'date' analysis-date marker column is available, same as NRTBreakpoint's
    own output.
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


def merge_drained_rows(df, target_month, drain_threshold, breaks_file):
    """Append this month's drained lakes into the running nrt_monthly_drain_breaks.parquet.

    Mirrors water_timeseries.scripts.merge_nrt_confidence.merge_nrt_confidence:
    back up the existing table, drop this month's existing rows (safe to
    re-run), concat the new ones in. Uses drained_mask (water_timeseries.utils.
    nrt_postprocessing) rather than a plain water_residual comparison, since it
    also accounts for drainage_confidence when present.
    """
    drained = df[drained_mask(df, drain_threshold=drain_threshold)].copy()
    drained.insert(1, "analysis_month", target_month)
    keep_cols = [c for c in ("id_geohash", "analysis_month", *NRT_MONTHLY_TILE_PROPERTIES) if c in drained.columns]
    keep_cols = list(dict.fromkeys(keep_cols))
    drained = drained[keep_cols]

    logger.info(f"{target_month}: {len(drained):,}/{len(df):,} lakes classified as drained (drain_threshold={drain_threshold})")

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
    drain_threshold = float(os.environ.get("drain_threshold", DRAIN_THRESHOLD))
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

    breaks_file = nrt_precomputed_dir / "nrt_monthly_drain_breaks.parquet"
    drained = merge_drained_rows(df, target_month, drain_threshold, breaks_file)

    if drained.empty:
        logger.warning(f"No drained lakes for {target_month} - skipping pmtiles build "
                        f"(the base archive already covers every stable lake)")
        return {'success': True, 'target_month': target_month, 'pmtiles': {}}

    outputs = build_pmtiles_nrt_monthly(
        breaks_parquet=breaks_file,
        geometry_parquet=vector_lake_file,
        output_dir=nrt_pmtiles_output_dir,
        months=[target_month],
        poly_max_zoom=poly_max_zoom,
        drain_threshold=drain_threshold,
    )

    for month, path in outputs.items():
        logger.success(f"[{month}] wrote {path}")

    return {
        'success': True,
        'target_month': target_month,
        'pmtiles': {month: str(path) for month, path in outputs.items()},
    }


if __name__ == "__main__":
    main()
