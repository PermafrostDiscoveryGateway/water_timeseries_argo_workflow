"""
Compare zarr archives between two output directories.

Walks every *.zarr archive found under `output_dir_test`, finds the
corresponding archive at the same relative path under `output_dir`, and
prints a summary comparison (sizes, dimensions, and per-variable
differences where possible).

Usage: python compare_results.py [path_to_env]
"""

import nest_asyncio
# Zarr v3's async I/O keeps a per-thread event loop; running under a debugger
# (e.g. PyCharm's pydevd) can hand a later store-open a Future tied to a stale
# loop ("Task ... got Future ... attached to a different loop"). Patching
# asyncio to tolerate re-entrant loops here avoids that crash.
nest_asyncio.apply()

from dotenv import load_dotenv
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import xarray as xr

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def print_section(title, char='='):
    print(f"\n{char * 80}")
    print(f"{title}")
    print(f"{char * 80}")


def get_dir_size_bytes(path: str) -> int:
    """Total size in bytes of every file under a zarr archive directory."""
    total = 0
    for f in Path(path).rglob('*'):
        if f.is_file():
            total += f.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    size_gb = size_bytes / (1024 ** 3)
    if size_gb >= 1:
        return f"{size_gb:.2f} GB"
    size_mb = size_bytes / (1024 ** 2)
    if size_mb >= 1:
        return f"{size_mb:.2f} MB"
    return f"{size_bytes / 1024:.2f} KB"


def find_zarr_archives(base_dir: str):
    """Return every *.zarr archive path under base_dir, relative to base_dir."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    return sorted(
        str(p.relative_to(base_path))
        for p in base_path.rglob('*.zarr')
        if p.is_dir()
    )


def compare_dataset_variables(ds_test: xr.Dataset, ds_other: xr.Dataset) -> dict:
    """Compare variables/values between two opened zarr datasets. Best-effort - any
    mismatch in shape/dtype is reported rather than raising."""
    comparison = {}

    test_vars = set(ds_test.data_vars)
    other_vars = set(ds_other.data_vars)

    comparison['vars_only_in_test'] = sorted(test_vars - other_vars)
    comparison['vars_only_in_other'] = sorted(other_vars - test_vars)

    shared_vars = sorted(test_vars & other_vars)
    comparison['variables'] = {}

    for var in shared_vars:
        var_test = ds_test[var]
        var_other = ds_other[var]

        if var_test.shape != var_other.shape:
            comparison['variables'][var] = {
                'shapes_match': False,
                'shape_test': var_test.shape,
                'shape_other': var_other.shape,
            }
            continue

        try:
            values_test = var_test.values
            values_other = var_other.values

            if np.issubdtype(values_test.dtype, np.number):
                diff_mask = ~np.isclose(
                    values_test.astype(float), values_other.astype(float),
                    equal_nan=True
                )
                n_diff = int(np.sum(diff_mask))
                comparison['variables'][var] = {
                    'shapes_match': True,
                    'identical': n_diff == 0,
                    'n_values': values_test.size,
                    'n_differing': n_diff,
                    'pct_differing': (n_diff / values_test.size * 100) if values_test.size else 0.0,
                }
            else:
                n_diff = int(np.sum(values_test.astype(str) != values_other.astype(str)))
                comparison['variables'][var] = {
                    'shapes_match': True,
                    'identical': n_diff == 0,
                    'n_values': values_test.size,
                    'n_differing': n_diff,
                    'pct_differing': (n_diff / values_test.size * 100) if values_test.size else 0.0,
                }
        except Exception as e:
            comparison['variables'][var] = {'shapes_match': True, 'error': str(e)}

    return comparison


def compare_zarr_archive(rel_path: str, test_path: str, other_path: str) -> dict:
    print_section(f"COMPARING: {rel_path}")
    print(f"Test archive:  {test_path}")
    print(f"Other archive: {other_path}")

    result = {'rel_path': rel_path, 'test_path': test_path, 'other_path': other_path}

    if not os.path.exists(other_path):
        print("Corresponding archive not found in the other output dir")
        result['found_in_other'] = False
        return result
    result['found_in_other'] = True

    # ---- Size comparison ----
    size_test = get_dir_size_bytes(test_path)
    size_other = get_dir_size_bytes(other_path)
    same_size = size_test == size_other

    print(f"\nSize (test):  {format_size(size_test)}")
    print(f"Size (other): {format_size(size_other)}")
    print(f"Same size: {same_size}")

    result['size_test_bytes'] = size_test
    result['size_other_bytes'] = size_other
    result['same_size'] = same_size

    # ---- Detailed comparison ----
    try:
        ds_test = xr.open_zarr(test_path)
        ds_other = xr.open_zarr(other_path)
    except Exception as e:
        print(f"Could not open one or both archives for detailed comparison: {e}")
        result['detailed_comparison'] = False
        result['error'] = str(e)
        return result

    result['detailed_comparison'] = True

    print(f"\nDimensions (test):  {dict(ds_test.dims)}")
    print(f"Dimensions (other): {dict(ds_other.dims)}")
    dims_match = dict(ds_test.dims) == dict(ds_other.dims)
    print(f"Same dimensions: {dims_match}")
    result['dims_match'] = dims_match

    if 'date' in ds_test.coords and 'date' in ds_other.coords:
        dates_test = set(pd.to_datetime(ds_test.date.values))
        dates_other = set(pd.to_datetime(ds_other.date.values))
        if dates_test != dates_other:
            print(f"Dates only in test:  {sorted(dates_test - dates_other)}")
            print(f"Dates only in other: {sorted(dates_other - dates_test)}")

    var_comparison = compare_dataset_variables(ds_test, ds_other)
    result['variable_comparison'] = var_comparison

    if var_comparison['vars_only_in_test']:
        print(f"\nVariables only in test:  {var_comparison['vars_only_in_test']}")
    if var_comparison['vars_only_in_other']:
        print(f"Variables only in other: {var_comparison['vars_only_in_other']}")

    print("\nPer-variable comparison:")
    for var, stats in var_comparison['variables'].items():
        if stats.get('error'):
            print(f"  {var}: error comparing - {stats['error']}")
        elif not stats.get('shapes_match', True):
            print(f"  {var}: shape mismatch - test={stats['shape_test']} other={stats['shape_other']}")
        elif stats.get('identical'):
            print(f"  {var}: identical ({stats['n_values']:,} values)")
        else:
            print(
                f"  {var}: {stats['n_differing']:,}/{stats['n_values']:,} values differ "
                f"({stats['pct_differing']:.2f}%)"
            )

    ds_test.close()
    ds_other.close()

    return result


def main():
    logger.debug("Comparing zarr archives between output_dir_test and output_dir")

    env_path = None
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    output_dir_test = os.environ.get('output_dir_test')
    output_dir = os.environ.get('output_dir')

    if not output_dir_test:
        logger.error("output_dir_test not set in environment")
        return {'success': False, 'error': 'output_dir_test not set'}
    if not output_dir:
        logger.error("output_dir not set in environment")
        return {'success': False, 'error': 'output_dir not set'}

    print_section("ZARR RESULTS COMPARISON")
    print(f"output_dir_test: {output_dir_test}")
    print(f"output_dir:      {output_dir}")

    zarr_archives = find_zarr_archives(output_dir_test)
    if not zarr_archives:
        logger.error(f"No .zarr archives found under {output_dir_test}")
        return {'success': False, 'error': 'No .zarr archives found in output_dir_test'}

    print(f"\nFound {len(zarr_archives)} zarr archive(s) under output_dir_test")

    results = []
    for rel_path in zarr_archives:
        test_path = os.path.join(output_dir_test, rel_path)
        other_path = os.path.join(output_dir, rel_path)
        results.append(compare_zarr_archive(rel_path, test_path, other_path))

    # ---- Overall summary ----
    print_section("SUMMARY")
    n_total = len(results)
    n_missing = sum(1 for r in results if not r.get('found_in_other'))
    n_same_size = sum(1 for r in results if r.get('same_size'))
    n_dims_match = sum(1 for r in results if r.get('dims_match'))

    print(f"Total archives compared: {n_total}")
    print(f"Missing in output_dir:   {n_missing}")
    print(f"Same size:               {n_same_size}/{n_total - n_missing}")
    print(f"Same dimensions:         {n_dims_match}/{n_total - n_missing}")

    for r in results:
        if not r.get('found_in_other'):
            print(f"  ❌ MISSING: {r['rel_path']}")
        elif r.get('same_size') and r.get('dims_match'):
            print(f"  ✅ MATCH:   {r['rel_path']}")
        else:
            print(f"  ⚠️  DIFFERS: {r['rel_path']}")

    return {'success': True, 'results': results}


if __name__ == "__main__":
    main()
