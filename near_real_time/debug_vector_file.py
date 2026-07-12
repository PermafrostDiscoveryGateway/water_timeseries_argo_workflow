from near_real_time_grid_v2 import verify_downloads_complete, verify_process_complete, merge_near_real_time_region, \
    process_near_real_time_region_dates_zarr, download_near_real_time_region_dates, generate_expected_dates, \
    merge_near_real_time_region_v3_simple, find_matching_lake_ids, \
    compare_netcdf_files, verify_merged_netcdf, verify_merged_data, merge_new_results, is_all_new_data_in_file
import sys
import utils.download_new_dynamic_world_data as download_new_dynamic_world_data
from loguru import logger
from datetime import date, datetime
from dotenv import load_dotenv
import os
import glob
import time
import pandas as pd
import utils.region_boundaries
from pathlib import Path
import xarray as xr
import numpy as np
import shutil
import gc
from typing import List, Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def compare_vector_files(
        old_file_path: str,
        new_file_path: str,
        region: Optional[str] = None,
        env_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare old and new vector files to understand differences.

    Args:
        old_file_path: Path to the old vector file
        new_file_path: Path to the new vector file
        region: Optional region name to filter by
        env_path: Optional path to .env file

    Returns:
        dict: Comprehensive comparison results
    """
    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    print(f"{'=' * 80}")
    print("COMPARING VECTOR FILES")
    print(f"{'=' * 80}")
    print(f"Old file: {old_file_path}")
    print(f"New file: {new_file_path}")

    result = {
        'old_file': old_file_path,
        'new_file': new_file_path,
        'region': region,
        'old_file_exists': Path(old_file_path).exists(),
        'new_file_exists': Path(new_file_path).exists(),
        'comparison': {}
    }

    if not result['old_file_exists']:
        result['error'] = f'Old file not found: {old_file_path}'
        return result

    if not result['new_file_exists']:
        result['error'] = f'New file not found: {new_file_path}'
        return result

    try:
        import geopandas as gpd
        import pandas as pd
        import numpy as np

        # ========== LOAD BOTH FILES ==========
        print("\n📁 Loading files...")
        gdf_old = gpd.read_parquet(old_file_path)
        gdf_new = gpd.read_parquet(new_file_path)

        print(f"  Old file: {len(gdf_old):,} rows")
        print(f"  New file: {len(gdf_new):,} rows")

        # ========== BASIC INFO COMPARISON ==========
        print("\n📊 BASIC INFO:")
        print(f"  Old columns: {gdf_old.columns.tolist()}")
        print(f"  New columns: {gdf_new.columns.tolist()}")

        old_geom_types = gdf_old.geometry.geom_type.unique()
        new_geom_types = gdf_new.geometry.geom_type.unique()
        print(f"  Old geometry types: {old_geom_types}")
        print(f"  New geometry types: {new_geom_types}")

        print(f"  Old CRS: {gdf_old.crs}")
        print(f"  New CRS: {gdf_new.crs}")

        # ========== ID COLUMN COMPARISON ==========
        id_col_candidates = ['id_geohash', 'id', 'geohash']
        old_id_col = None
        new_id_col = None

        for col in id_col_candidates:
            if col in gdf_old.columns:
                old_id_col = col
                break

        for col in id_col_candidates:
            if col in gdf_new.columns:
                new_id_col = col
                break

        if old_id_col:
            print(f"\n🆔 Old ID column: {old_id_col}")
            print(f"  Type: {gdf_old[old_id_col].dtype}")
            print(f"  Sample: {gdf_old[old_id_col].head(3).tolist()}")
            print(f"  Unique count: {gdf_old[old_id_col].nunique():,}")
        else:
            print(f"\n⚠️ No ID column found in old file!")

        if new_id_col:
            print(f"\n🆔 New ID column: {new_id_col}")
            print(f"  Type: {gdf_new[new_id_col].dtype}")
            print(f"  Sample: {gdf_new[new_id_col].head(3).tolist()}")
            print(f"  Unique count: {gdf_new[new_id_col].nunique():,}")
        else:
            print(f"\n⚠️ No ID column found in new file!")

        # ========== GEOMETRY COMPARISON ==========
        print("\n🌍 GEOMETRY COMPARISON:")

        # Handle Polygon geometries - convert to centroids for coordinate access
        def get_coordinates(gdf, geom_col='geometry'):
            geom_type = gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else None

            if geom_type in ['Polygon', 'MultiPolygon']:
                centroids = gdf.geometry.centroid
                return centroids.x, centroids.y
            elif geom_type == 'Point':
                return gdf.geometry.x, gdf.geometry.y
            else:
                rep_points = gdf.geometry.representative_point()
                return rep_points.x, rep_points.y

        old_x, old_y = get_coordinates(gdf_old)
        new_x, new_y = get_coordinates(gdf_new)

        print(f"  Old coordinate range:")
        print(f"    Longitude: {old_x.min():.2f} to {old_x.max():.2f}")
        print(f"    Latitude: {old_y.min():.2f} to {old_y.max():.2f}")

        print(f"  New coordinate range:")
        print(f"    Longitude: {new_x.min():.2f} to {new_x.max():.2f}")
        print(f"    Latitude: {new_y.min():.2f} to {new_y.max():.2f}")

        # ========== REGION COMPARISON ==========
        print("\n📍 REGION COMPARISON:")

        if 'region' in gdf_old.columns:
            print(f"  Old has region column")
            old_regions = gdf_old['region'].unique()
            print(f"  Old regions: {sorted(old_regions)}")
            for reg in sorted(old_regions):
                count = len(gdf_old[gdf_old['region'] == reg])
                print(f"    {reg}: {count:,} lakes")
        else:
            print("  Old has NO region column")

        if 'region' in gdf_new.columns:
            print(f"  New has region column")
            new_regions = gdf_new['region'].unique()
            print(f"  New regions: {sorted(new_regions)}")
            for reg in sorted(new_regions):
                count = len(gdf_new[gdf_new['region'] == reg])
                print(f"    {reg}: {count:,} lakes")
        else:
            print("  New has NO region column")

        # ========== ID OVERLAP COMPARISON ==========
        if old_id_col and new_id_col:
            print("\n🔗 ID OVERLAP:")
            old_ids = set(gdf_old[old_id_col].values)
            new_ids = set(gdf_new[new_id_col].values)

            print(f"  Old unique IDs: {len(old_ids):,}")
            print(f"  New unique IDs: {len(new_ids):,}")

            overlap = old_ids.intersection(new_ids)
            old_only = old_ids - new_ids
            new_only = new_ids - old_ids

            print(f"  Overlapping IDs: {len(overlap):,}")
            print(f"  IDs only in old: {len(old_only):,}")
            print(f"  IDs only in new: {len(new_only):,}")

            # Check if ID types match
            old_sample = next(iter(old_ids)) if old_ids else None
            new_sample = next(iter(new_ids)) if new_ids else None

            print(f"\n  ID type check:")
            print(f"    Old ID type: {type(old_sample)}")
            print(f"    New ID type: {type(new_sample)}")
            print(f"    Types match: {type(old_sample) == type(new_sample)}")

            if old_sample and new_sample:
                print(f"    Old ID example: {old_sample}")
                print(f"    New ID example: {new_sample}")

        # ========== REGION-SPECIFIC FILTERING ==========
        if region:
            print(f"\n📌 REGION-SPECIFIC FILTERING: {region}")

            # Get region boundaries
            from utils.region_boundaries import get_region_boundaries
            boundaries = get_region_boundaries()

            if region in boundaries:
                bbox = boundaries[region]
                print(f"  Boundaries: {bbox}")

                # Filter old file
                mask_old = (old_x >= bbox['X_MIN_START']) & (old_x <= bbox['X_MIN_END']) & \
                           (old_y >= bbox['Y_MIN_START']) & (old_y <= bbox['Y_MIN_END'])
                old_in_region = gdf_old[mask_old]

                # Filter new file
                mask_new = (new_x >= bbox['X_MIN_START']) & (new_x <= bbox['X_MIN_END']) & \
                           (new_y >= bbox['Y_MIN_START']) & (new_y <= bbox['Y_MIN_END'])
                new_in_region = gdf_new[mask_new]

                print(f"\n  Lakes in {region} boundaries:")
                print(f"    Old: {len(old_in_region):,}")
                print(f"    New: {len(new_in_region):,}")

                # Check ID overlap within region
                if old_id_col and new_id_col and len(old_in_region) > 0 and len(new_in_region) > 0:
                    old_region_ids = set(old_in_region[old_id_col].values)
                    new_region_ids = set(new_in_region[new_id_col].values)
                    overlap_region = old_region_ids.intersection(new_region_ids)

                    print(f"\n  ID overlap within {region}:")
                    print(f"    Old region IDs: {len(old_region_ids):,}")
                    print(f"    New region IDs: {len(new_region_ids):,}")
                    print(f"    Overlapping: {len(overlap_region):,}")
                    print(f"    IDs only in old: {len(old_region_ids - new_region_ids):,}")
                    print(f"    IDs only in new: {len(new_region_ids - old_region_ids):,}")

                    if len(overlap_region) > 0:
                        print(f"    Sample overlapping IDs: {list(overlap_region)[:5]}")
            else:
                print(f"  ⚠️ Region {region} not found in boundaries!")

        # ========== COMPARE WITH DATA FILES ==========
        print("\n📊 COMPARING WITH DATA FILES:")

        dynamic_world_data_dir = os.environ.get('dynamic_world_data')
        if dynamic_world_data_dir:
            date_to_check = "2026-06"  # Or pass this as a parameter
            data_file = Path(dynamic_world_data_dir) / 'merge' / f'dw_{region}_{date_to_check}.nc' if region else None

            if data_file and data_file.exists():
                try:
                    ds = xr.open_dataset(str(data_file))
                    data_ids = set(ds.id_geohash.values)
                    ds.close()

                    print(f"  Data file: {data_file}")
                    print(f"  IDs in data file: {len(data_ids):,}")

                    if old_id_col:
                        old_ids_set = set(gdf_old[old_id_col].values)
                        overlap_old = data_ids.intersection(old_ids_set)
                        print(f"  Overlap old vs data: {len(overlap_old):,}")

                    if new_id_col:
                        new_ids_set = set(gdf_new[new_id_col].values)
                        overlap_new = data_ids.intersection(new_ids_set)
                        print(f"  Overlap new vs data: {len(overlap_new):,}")

                    if old_id_col and new_id_col:
                        old_only_data = data_ids - set(gdf_old[old_id_col].values)
                        new_only_data = data_ids - set(gdf_new[new_id_col].values)
                        print(f"  IDs in data but not old: {len(old_only_data):,}")
                        print(f"  IDs in data but not new: {len(new_only_data):,}")

                except Exception as e:
                    print(f"  Error checking data file: {e}")
            else:
                print(f"  No data file found for {region} {date_to_check}")

        # ========== SAMPLE COMPARISON ==========
        print("\n📋 SAMPLE COMPARISON (first 5 rows):")
        print("  Old file:")
        if old_id_col:
            print(gdf_old[[old_id_col, 'geometry']].head(5))
        else:
            print(gdf_old.head(5))

        print("\n  New file:")
        if new_id_col:
            print(gdf_new[[new_id_col, 'geometry']].head(5))
        else:
            print(gdf_new.head(5))

        result['comparison'] = {
            'old_count': len(gdf_old),
            'new_count': len(gdf_new),
            'old_columns': gdf_old.columns.tolist(),
            'new_columns': gdf_new.columns.tolist(),
            'old_geom_types': list(old_geom_types),
            'new_geom_types': list(new_geom_types),
            'old_coords': {'x_min': float(old_x.min()), 'x_max': float(old_x.max()),
                           'y_min': float(old_y.min()), 'y_max': float(old_y.max())},
            'new_coords': {'x_min': float(new_x.min()), 'x_max': float(new_x.max()),
                           'y_min': float(new_y.min()), 'y_max': float(new_y.max())},
        }

        if old_id_col and new_id_col:
            result['comparison']['id_overlap'] = {
                'overlap_count': len(overlap),
                'old_only_count': len(old_only),
                'new_only_count': len(new_only),
                'overlap_sample': list(overlap)[:10],
                'old_only_sample': list(old_only)[:10],
                'new_only_sample': list(new_only)[:10],
            }

        if region and region in boundaries:
            result['comparison']['region_specific'] = {
                'old_in_region': len(old_in_region),
                'new_in_region': len(new_in_region),
                'old_ids_in_region': len(old_region_ids) if old_id_col and len(old_in_region) > 0 else 0,
                'new_ids_in_region': len(new_region_ids) if new_id_col and len(new_in_region) > 0 else 0,
                'overlap_in_region': len(overlap_region) if old_id_col and new_id_col and len(
                    old_in_region) > 0 and len(new_in_region) > 0 else 0,
            }

        print("\n✅ Comparison complete!")
        return result

    except Exception as e:
        print(f"❌ Error comparing files: {e}")
        import traceback
        traceback.print_exc()
        result['error'] = str(e)
        return result


def debug_vector_file(region: str = "EURASIA3", env_path: str = None, use_new_vector: bool = False):
    """
    Debug the vector file to understand why lakes aren't being found for a region.
    Can optionally use the new vector file for comparison.
    """
    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    vector_file = os.environ.get('vector_lake_file')
    if not vector_file:
        print("❌ vector_lake_file not set in environment")
        return

    print(f"{'=' * 70}")
    print(f"DEBUGGING VECTOR FILE FOR REGION: {region}")
    print(f"{'=' * 70}")
    print(f"Vector file: {vector_file}")

    try:
        # Load the vector file
        import geopandas as gpd
        import xarray as xr
        import pandas as pd
        import numpy as np

        gdf = gpd.read_parquet(vector_file)

        # Handle Polygon geometries - convert to centroids for coordinate access
        geom_type = gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else None

        if geom_type in ['Polygon', 'MultiPolygon']:
            print("⚠️ Converting Polygon geometries to centroids for coordinate access")
            gdf['centroid'] = gdf.geometry.centroid
            gdf = gdf.set_geometry('centroid')

        print(f"\n📊 VECTOR FILE INFO:")
        print(f"  Total lakes: {len(gdf)}")
        print(f"  Columns: {gdf.columns.tolist()}")
        print(f"  Geometry type: {geom_type}")
        print(f"  CRS: {gdf.crs}")

        # Check coordinate ranges
        print(f"\n🌍 COORDINATE RANGES:")
        print(f"  Longitude: {gdf.geometry.x.min():.2f} to {gdf.geometry.x.max():.2f}")
        print(f"  Latitude: {gdf.geometry.y.min():.2f} to {gdf.geometry.y.max():.2f}")

        # Check ID column
        if 'id_geohash' in gdf.columns:
            id_col = 'id_geohash'
        elif 'id' in gdf.columns:
            id_col = 'id'
        else:
            print(f"  ⚠️ No ID column found! Available: {gdf.columns.tolist()}")
            id_col = None

        if id_col:
            print(f"\n🆔 ID COLUMN: {id_col}")
            print(f"  Type: {gdf[id_col].dtype}")
            print(f"  Example IDs: {gdf[id_col].head(5).tolist()}")
            print(f"  Total unique IDs: {gdf[id_col].nunique()}")

        # Check region information
        if 'region' in gdf.columns:
            print(f"\n📍 REGIONS IN VECTOR FILE:")
            for reg in sorted(gdf['region'].unique()):
                count = len(gdf[gdf['region'] == reg])
                print(f"  {reg}: {count} lakes")
        else:
            print(f"\n⚠️ No 'region' column found in vector file")
            print(f"  Available columns: {gdf.columns.tolist()}")

            # Try to filter by bounding box instead
            region_boundaries = {
                'TEST': {'Y_MIN_START': 62, 'Y_MIN_END': 64, 'X_MIN_START': 153, 'X_MIN_END': 156},
                'ALASKA': {'Y_MIN_START': 55, 'Y_MIN_END': 72, 'X_MIN_START': -168, 'X_MIN_END': -138},
                'CANADA': {'Y_MIN_START': 50, 'Y_MIN_END': 80, 'X_MIN_START': -141, 'X_MIN_END': -54},
                'EURASIA1': {'Y_MIN_START': 55, 'Y_MIN_END': 71, 'X_MIN_START': 18, 'X_MIN_END': 63},
                'EURASIA2': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': 66, 'X_MIN_END': 177},
                'EURASIA3': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': -180, 'X_MIN_END': -169},
            }

            bbox = region_boundaries.get(region)
            if bbox:
                # Filter by bounding box
                in_bbox = gdf[
                    (gdf.geometry.x >= bbox['X_MIN_START']) &
                    (gdf.geometry.x <= bbox['X_MIN_END']) &
                    (gdf.geometry.y >= bbox['Y_MIN_START']) &
                    (gdf.geometry.y <= bbox['Y_MIN_END'])
                    ]
                print(f"\n  Lakes in {region} bounding box: {len(in_bbox)}")
                if len(in_bbox) > 0:
                    print(f"  Sample IDs in bounding box: {in_bbox[id_col].head(5).tolist()}")

        # Check if the region's IDs exist in the vector file
        # Load new data file to get region IDs
        dynamic_world_data_dir = os.environ.get('dynamic_world_data')
        if dynamic_world_data_dir:
            new_data_file = f"{dynamic_world_data_dir}/merge/dw_{region}_2026-06.nc"
            try:
                ds = xr.open_dataset(new_data_file)
                data_ids = set(ds.id_geohash.values)
                ds.close()
                print(f"\n📊 NEW DATA FILE CHECK:")
                print(f"  New data file: {new_data_file}")
                print(f"  IDs in new data: {len(data_ids)}")

                if id_col and len(data_ids) > 0:
                    vector_ids = set(gdf[id_col].values)
                    overlap = data_ids.intersection(vector_ids)
                    print(f"  Overlapping IDs between new data and vector: {len(overlap)}")

                    if len(overlap) == 0:
                        print(f"\n⚠️ NO OVERLAPPING IDs FOUND!")
                        print(f"  Data ID type: {type(next(iter(data_ids)))}")
                        print(f"  Vector ID type: {type(next(iter(vector_ids)))}")
                        print(f"  Data ID example: {next(iter(data_ids))}")
                        print(f"  Vector ID example: {next(iter(vector_ids))}")
            except Exception as e:
                print(f"  Error loading new data file: {e}")

        # Show sample of vector file
        print(f"\n📋 SAMPLE OF VECTOR FILE:")
        print(gdf[['id_geohash', 'geometry']].head(10))

    except Exception as e:
        print(f"❌ Error debugging vector file: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# IMPLEMENTATION OF MERGE_NEW_RESULTS
# =============================================================================
def merge_new_results(
        region: str = 'TEST',
        date_to_merge: str = None,
        merged_file_path: str = None,
        env_path: str = None
) -> Dict[str, Any]:
    """
    Merge new downloaded results for a single date into a NetCDF file.

    This function:
    1. Finds all downloaded files for the specified date
    2. Combines them into a single dataset
    3. Saves the combined data to the specified file path

    Args:
        region: Region name (e.g., "TEST", "AFRICA")
        date_to_merge: Date in "YYYY-MM" format
        merged_file_path: Path where the merged file should be saved
        env_path: Optional path to .env file

    Returns:
        dict: Result with status, file path, and statistics
    """
    logger.debug(f"Merging new results for {region} and {date_to_merge} into file {merged_file_path}")

    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Validate inputs
    if date_to_merge is None:
        logger.error("date_to_merge is required")
        return {'success': False, 'error': 'date_to_merge is required'}

    if merged_file_path is None:
        dynamic_world_data_dir = os.environ.get('dynamic_world_data')
        if not dynamic_world_data_dir:
            logger.error("dynamic_world_data not set in environment")
            return {'success': False, 'error': 'dynamic_world_data not set'}
        merged_file_path = os.path.join(dynamic_world_data_dir, f"dw_{region}_{date_to_merge}.nc")

    # Ensure directory exists
    Path(merged_file_path).parent.mkdir(parents=True, exist_ok=True)

    # Find downloaded files for this date
    dynamic_world_download_dir = Path(os.environ.get('dynamic_world_downloads', ''))
    download_dir = dynamic_world_download_dir / region / f'download_{date_to_merge}'

    if not download_dir.exists():
        logger.warning(f"Download directory does not exist: {download_dir}")
        return {'success': False, 'error': f'Download directory not found: {download_dir}'}

    # Get all downloaded NetCDF files for this date
    downloaded_files = sorted(glob.glob(str(download_dir / f'DW_{date_to_merge}_*.nc')))

    if not downloaded_files:
        logger.warning(f"No downloaded files found for {date_to_merge} in {download_dir}")
        return {'success': False, 'error': f'No downloaded files found for {date_to_merge}'}

    logger.info(f"Found {len(downloaded_files)} downloaded files for {date_to_merge}")

    try:
        # Combine all downloaded files into a single dataset
        logger.info("Combining downloaded files...")
        combined = None
        failed_files = []

        for nc_file in downloaded_files:
            try:
                ds = xr.open_dataset(nc_file)
                if len(ds['id_geohash']) > 0:
                    if combined is None:
                        combined = ds
                    else:
                        # Concatenate along id_geohash dimension
                        combined = xr.concat([combined, ds], dim='id_geohash')
                        # Remove duplicate IDs
                        _, unique_idx = np.unique(combined['id_geohash'].values, return_index=True)
                        if len(unique_idx) < len(combined['id_geohash']):
                            combined = combined.isel(id_geohash=np.sort(unique_idx))
                else:
                    logger.warning(f"File {nc_file} has no IDs, skipping")
                    failed_files.append(nc_file)
            except Exception as e:
                logger.error(f"Error opening {nc_file}: {e}")
                failed_files.append(nc_file)

        if combined is None:
            logger.error("No valid data to merge")
            return {'success': False, 'error': 'No valid data to merge'}

        logger.info(f"Combined dataset has {len(combined['id_geohash'])} IDs and {len(combined['date'])} dates")

        # Ensure date dimension is correct
        if len(combined['date']) > 0:
            # Sort by date
            combined = combined.sortby('date')

        # Write to NetCDF with compression
        logger.info(f"Writing merged data to {merged_file_path}")

        encoding = {}
        for var in combined.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 4,
                'shuffle': True
            }

        # Write to file
        combined.to_netcdf(merged_file_path, encoding=encoding)

        # Get file size
        file_size_gb = Path(merged_file_path).stat().st_size / (1024 ** 3)

        # Clean up
        combined.close()
        gc.collect()

        result = {
            'success': True,
            'file_path': merged_file_path,
            'id_count': len(combined['id_geohash']),
            'date_count': len(combined['date']),
            'file_size_gb': round(file_size_gb, 4),
            'files_merged': len(downloaded_files) - len(failed_files),
            'files_failed': len(failed_files),
            'failed_files': failed_files if failed_files else None,
            'region': region,
            'date': date_to_merge
        }

        logger.info(f"✅ Merge completed successfully!")
        logger.info(f"  File: {merged_file_path}")
        logger.info(f"  IDs: {result['id_count']:,}")
        logger.info(f"  Dates: {result['date_count']}")
        logger.info(f"  Size: {result['file_size_gb']:.4f} GB")

        return result

    except Exception as e:
        logger.error(f"Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


# =============================================================================
# FAST VECTORIZED VERIFICATION
# =============================================================================
def verify_region_data_vectorized(
        region: str,
        date_to_check: str,
        file_path: str,
        env_path: str = None,
        sample_size: int = 1000  # Number of IDs to sample for checking
) -> Dict[str, Any]:
    """
    Fast vectorized verification of region data in a file.
    Uses sampling for large regions instead of checking every ID.
    """
    logger.debug(f"Fast vectorized check for {region} and {date_to_check} in {file_path}")

    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Check if file exists
    if not Path(file_path).exists():
        return {'success': False, 'error': 'File not found', 'file_exists': False}

    try:
        ds = xr.open_dataset(file_path)

        # Check date presence
        dates_in_file = pd.to_datetime(ds['date'].values)
        date_strings = [d.strftime("%Y-%m") for d in dates_in_file]
        date_present = date_to_check in date_strings

        if not date_present:
            ds.close()
            return {'success': False, 'date_present': False, 'error': f'Date {date_to_check} not found'}

        # Get region IDs from vector file
        from utils.region_boundaries import get_region_boundaries
        region_boundaries = get_region_boundaries()

        if region not in region_boundaries:
            ds.close()
            return {'success': False, 'error': f'Region {region} not found in boundaries'}

        vector_lake_file = os.environ.get('vector_lake_file')
        if not vector_lake_file or not Path(vector_lake_file).exists():
            ds.close()
            return {'success': False, 'error': 'Vector lake file not found'}

        import geopandas as gpd
        gdf = gpd.read_parquet(vector_lake_file)

        bounds = region_boundaries[region]
        x_min_start = bounds['X_MIN_START']
        x_min_end = bounds['X_MIN_END']
        y_min_start = bounds['Y_MIN_START']
        y_min_end = bounds['Y_MIN_END']

        # Handle geometry types
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

        # Filter by bounding box
        mask = (x_coords >= x_min_start) & (x_coords <= x_min_end) & \
               (y_coords >= y_min_start) & (y_coords <= y_min_end)

        gdf_subset = gdf[mask]
        region_ids = gdf_subset['id_geohash'].values.tolist()

        if not region_ids:
            ds.close()
            return {'success': False, 'error': f'No IDs found for region {region}'}

        total_region_ids = len(region_ids)
        logger.info(f"Region {region} has {total_region_ids:,} IDs")

        # Get IDs in file
        file_ids = set(ds['id_geohash'].values)

        # Check if all region IDs are in file (using set operations - fast)
        region_ids_set = set(region_ids)
        missing_ids = region_ids_set - file_ids
        ids_in_file = region_ids_set & file_ids

        logger.info(f"IDs in file: {len(ids_in_file):,}, IDs missing: {len(missing_ids):,}")

        if len(missing_ids) > 0:
            ds.close()
            return {
                'success': False,
                'total_ids': total_region_ids,
                'ids_in_file': len(ids_in_file),
                'ids_missing': len(missing_ids),
                'missing_sample': list(missing_ids)[:10],
                'error': f'{len(missing_ids):,} IDs missing from file'
            }

        # Now check if data exists for the date (sample-based for speed)
        date_ts = pd.Timestamp(f"{date_to_check}-01")
        date_data = ds.sel(date=date_ts)

        # Find a data variable
        data_var = None
        for var_candidate in ['water', 'water_observed', 'water_predicted']:
            if var_candidate in date_data.data_vars:
                data_var = var_candidate
                break

        if data_var is None:
            ds.close()
            return {'success': True, 'date_present': True, 'warning': 'No data variable found'}

        # Sample-based check for data values
        ids_list = list(ids_in_file)
        sample_count = min(sample_size, len(ids_list))

        if sample_count < len(ids_list):
            # Sample IDs
            import random
            sampled_ids = random.sample(ids_list, sample_count)
            logger.info(f"Sampling {sample_count} IDs out of {len(ids_list):,} for data validation")

            ids_with_data = 0
            ids_without_data = 0

            for id_val in sampled_ids:
                try:
                    id_data = date_data.sel(id_geohash=id_val)
                    if data_var in id_data:
                        data_values = id_data[data_var].values
                        if np.any(~np.isnan(data_values)):
                            ids_with_data += 1
                        else:
                            ids_without_data += 1
                    else:
                        ids_without_data += 1
                except Exception:
                    ids_without_data += 1

            # If most sampled IDs have data, assume all do
            data_success_rate = ids_with_data / sample_count if sample_count > 0 else 0
            logger.info(f"Sample data success rate: {data_success_rate:.2%}")

            all_have_data = data_success_rate > 0.95  # 95% threshold

            ds.close()

            return {
                'success': all_have_data,
                'date_present': date_present,
                'total_ids': total_region_ids,
                'ids_in_file': len(ids_in_file),
                'ids_missing': 0,
                'sampled': sample_count,
                'ids_with_data': ids_with_data,
                'ids_without_data': ids_without_data,
                'data_success_rate': data_success_rate,
                'all_ids_have_data': all_have_data,
                'verification_method': 'sampled_vectorized'
            }
        else:
            # Small region - check all IDs
            logger.info(f"Checking all {len(ids_list):,} IDs for data validation")

            ids_with_data = []
            ids_without_data = []

            for id_val in ids_list:
                try:
                    id_data = date_data.sel(id_geohash=id_val)
                    if data_var in id_data:
                        data_values = id_data[data_var].values
                        if np.any(~np.isnan(data_values)):
                            ids_with_data.append(id_val)
                        else:
                            ids_without_data.append(id_val)
                    else:
                        ids_without_data.append(id_val)
                except Exception:
                    ids_without_data.append(id_val)

            all_have_data = len(ids_without_data) == 0 and len(ids_with_data) > 0

            ds.close()

            return {
                'success': all_have_data,
                'date_present': date_present,
                'total_ids': total_region_ids,
                'ids_in_file': len(ids_in_file),
                'ids_missing': 0,
                'ids_with_data': len(ids_with_data),
                'ids_without_data': len(ids_without_data),
                'all_ids_have_data': all_have_data,
                'verification_method': 'full_vectorized'
            }

    except Exception as e:
        logger.error(f"Error in vectorized verification: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


# =============================================================================
# COMBINE REGION FILES
# =============================================================================
def combine_region_files(
        region_files: List[str],
        output_file: str,
        env_path: str = None
) -> Dict[str, Any]:
    """
    Combine multiple region NetCDF files into a single combined file.
    """
    logger.info(f"\n{'=' * 80}")
    logger.info("COMBINING REGION FILES")
    logger.info(f"{'=' * 80}")
    logger.info(f"Number of files to combine: {len(region_files)}")
    logger.info(f"Output file: {output_file}")

    if not region_files:
        logger.error("No region files to combine")
        return {'success': False, 'error': 'No region files to combine'}

    # Verify all files exist
    missing_files = [f for f in region_files if not Path(f).exists()]
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        return {'success': False, 'error': f'Missing files: {missing_files}'}

    try:
        logger.info("Loading region datasets...")
        datasets = []
        file_info = []

        for file_path in region_files:
            try:
                ds = xr.open_dataset(file_path)
                id_count = len(ds['id_geohash']) if 'id_geohash' in ds.dims else 0
                date_count = len(ds['date']) if 'date' in ds.dims else 0
                file_size_gb = Path(file_path).stat().st_size / (1024 ** 3)

                file_info.append({
                    'file': file_path,
                    'id_count': id_count,
                    'date_count': date_count,
                    'file_size_gb': round(file_size_gb, 4)
                })

                datasets.append(ds)

            except Exception as e:
                logger.error(f"Error opening {file_path}: {e}")
                for ds in datasets:
                    try:
                        ds.close()
                    except:
                        pass
                return {'success': False, 'error': f'Error opening {file_path}: {e}'}

        logger.info("\nFiles to combine:")
        for info in file_info:
            logger.info(
                f"  {Path(info['file']).name}: {info['id_count']:,} IDs, {info['date_count']} dates, {info['file_size_gb']:.4f} GB")

        logger.info("Combining datasets...")
        combined = None

        for ds in datasets:
            if combined is None:
                combined = ds
            else:
                combined = xr.concat([combined, ds], dim='id_geohash')
                _, unique_idx = np.unique(combined['id_geohash'].values, return_index=True)
                if len(unique_idx) < len(combined['id_geohash']):
                    removed_count = len(combined['id_geohash']) - len(unique_idx)
                    logger.info(f"Removed {removed_count} duplicate IDs")
                    combined = combined.isel(id_geohash=np.sort(unique_idx))

        if combined is None:
            logger.error("No datasets to combine")
            return {'success': False, 'error': 'No datasets to combine'}

        combined = combined.sortby(['id_geohash', 'date'])

        logger.info(f"Combined dataset has {len(combined['id_geohash'])} IDs and {len(combined['date'])} dates")

        logger.info(f"Writing combined file to {output_file}")

        encoding = {}
        for var in combined.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 4,
                'shuffle': True
            }

        combined.to_netcdf(output_file, encoding=encoding)

        file_size_gb = Path(output_file).stat().st_size / (1024 ** 3)

        for ds in datasets:
            try:
                ds.close()
            except:
                pass
        combined.close()
        gc.collect()

        result = {
            'success': True,
            'file_path': output_file,
            'id_count': len(combined['id_geohash']),
            'date_count': len(combined['date']),
            'file_size_gb': round(file_size_gb, 4),
            'files_combined': len(datasets),
            'file_info': file_info
        }

        logger.info(f"✅ Combined file created successfully!")
        logger.info(f"  File: {output_file}")
        logger.info(f"  IDs: {result['id_count']:,}")
        logger.info(f"  Dates: {result['date_count']}")
        logger.info(f"  Size: {result['file_size_gb']:.4f} GB")

        return result

    except Exception as e:
        logger.error(f"Error combining files: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


# =============================================================================
# VERIFY COMBINED FILE (OPTIMIZED)
# =============================================================================
def verify_combined_file_optimized(
        combined_file_path: str,
        regions: List[str],
        date_to_check: str,
        env_path: str = None,
        sample_size: int = 1000
) -> Dict[str, Any]:
    """
    Verify the combined file contains all data for all regions.
    Uses vectorized + sampling for speed.
    """
    logger.info(f"\n{'=' * 80}")
    logger.info("VERIFYING COMBINED FILE (OPTIMIZED)")
    logger.info(f"{'=' * 80}")

    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    if not Path(combined_file_path).exists():
        return {'success': False, 'error': 'Combined file not found'}

    # First, do a quick check on the combined file itself
    try:
        combined_ds = xr.open_dataset(combined_file_path)
        combined_ids = set(combined_ds['id_geohash'].values)
        combined_dates = set(pd.to_datetime(combined_ds['date'].values))
        combined_date_strings = {d.strftime("%Y-%m") for d in combined_dates}

        logger.info(f"Combined file has {len(combined_ids):,} IDs and {len(combined_dates)} dates")

        # Check date
        date_present = date_to_check in combined_date_strings
        if not date_present:
            combined_ds.close()
            return {'success': False, 'error': f'Date {date_to_check} not found in combined file'}

        combined_ds.close()

    except Exception as e:
        logger.error(f"Error opening combined file: {e}")
        return {'success': False, 'error': str(e)}

    # Now check each region (fast vectorized checks)
    region_results = {}
    all_present = True
    total_ids = 0
    total_missing = 0

    for region in regions:
        logger.info(f"\nChecking region: {region}")

        # Use the fast vectorized verification
        result = verify_region_data_vectorized(
            region=region,
            date_to_check=date_to_check,
            file_path=combined_file_path,
            env_path=env_path,
            sample_size=sample_size
        )

        region_results[region] = result

        if result.get('success', False):
            logger.info(f"  ✅ Region {region} verified successfully")
        else:
            all_present = False
            error = result.get('error', 'Unknown error')
            logger.warning(f"  ❌ Region {region} failed: {error}")

            if 'ids_missing' in result:
                total_missing += result['ids_missing']

        total_ids += result.get('total_ids', 0)

    return {
        'success': all_present,
        'combined_file': combined_file_path,
        'total_ids': total_ids,
        'total_missing': total_missing,
        'all_regions_present': all_present,
        'region_results': region_results,
        'date_present': date_present
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_file_size_gb(file_path: str) -> float:
    """Get file size in GB."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 ** 3)
    return 0


def quick_check_merged_file(file_path: str, date_to_check: str) -> bool:
    """Quick check if a merged file exists and has the date."""
    if not Path(file_path).exists():
        return False
    try:
        ds = xr.open_dataset(file_path)
        dates_in_file = pd.to_datetime(ds['date'].values)
        date_strings = [d.strftime("%Y-%m") for d in dates_in_file]
        ds.close()
        return date_to_check in date_strings
    except:
        return False


def process_region_fast(
        region: str,
        date_to_run: str,
        env_path: str = None,
        dynamic_world_data_dir: str = None
) -> Dict[str, Any]:
    """
    Fast process a single region: verify downloads, merge (no verification).
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"PROCESSING REGION: {region} (FAST MODE - NO VERIFICATION)")
    logger.info(f"{'=' * 80}")

    result = {
        'region': region,
        'date': date_to_run,
        'success': False,
        'steps': {}
    }

    # Step 1: Verify downloads are complete (quick)
    logger.info(f"Step 1: Verifying downloads for {region}...")
    downloads_complete = verify_downloads_complete(region=region, analysis_dates=[date_to_run])

    complete = downloads_complete.get('complete', False)
    summary = downloads_complete.get('summary', {})

    total_expected = summary.get('total_expected_downloads', 0)
    total_skipped = summary.get('total_skipped_downloads', 0)
    total_successful = summary.get('total_successful_downloads', 0)
    total_skipped_and_successful = total_skipped + total_successful

    if total_expected > 0:
        percent_downloaded = float(total_skipped_and_successful) / float(total_expected)
        logger.info(f"  Percent downloaded: {percent_downloaded:.4f}")
        if percent_downloaded > 0.99:
            complete = True

    if not complete:
        logger.warning(f"⚠️ Downloads not complete for {region} - skipping")
        result['success'] = False
        result['reason'] = 'Downloads incomplete'
        return result

    # Step 2: Check if already merged (quick check)
    merged_file_path = os.path.join(dynamic_world_data_dir, f"dw_{region}_{date_to_run}.nc")

    if quick_check_merged_file(merged_file_path, date_to_run):
        logger.info(f"✅ Region {region} already has date {date_to_run} - skipping merge")
        result['success'] = True
        result['merged_file'] = merged_file_path
        result['reason'] = 'Already merged'
        return result

    # Step 3: Merge (no verification)
    logger.info(f"Step 2: Merging results for {region}...")
    merge_result = merge_new_results(
        region=region,
        date_to_merge=date_to_run,
        merged_file_path=merged_file_path,
        env_path=env_path
    )

    if not merge_result.get('success', False):
        logger.error(f"❌ Merge failed for {region}: {merge_result.get('error', 'Unknown error')}")
        result['success'] = False
        result['reason'] = 'Merge failed'
        return result

    logger.info(f"✅ Region {region} merged successfully (verification deferred)")
    result['success'] = True
    result['merged_file'] = merged_file_path
    result['reason'] = 'Successfully merged (verification deferred)'

    return result


# =============================================================================
# MAIN SCRIPT
# =============================================================================
def main():
    logger.debug(f"Beginning historical run for ALL regions")
    env_path = None
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    # ========== Compare vector files ==========
    old_vector_file = os.environ.get('vector_lake_file')
    new_vector_file = '/data/water_timeseries/vector_dataset_file/lake_polygons.parquet'

    def debug_eurasia3_specific(env_path: str = None):
        """Debug EURASIA3 specifically to find why lakes aren't being processed."""

        if env_path:
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()

        import geopandas as gpd
        import xarray as xr
        import pandas as pd

        # Load the correct vector file
        vector_file = os.environ.get('vector_lake_file')
        gdf = gpd.read_parquet(vector_file)

        # Handle Polygon geometries
        if 'Polygon' in gdf.geometry.geom_type.unique():
            gdf['centroid'] = gdf.geometry.centroid
            gdf = gdf.set_geometry('centroid')

        # Load EURASIA3 data file
        data_file = '/data/water_timeseries/dynamic_world_data/merge/dw_EURASIA3_2026-06.nc'
        ds = xr.open_dataset(data_file)
        data_ids = set(ds.id_geohash.values)
        ds.close()

        print(f"EURASIA3 data file has {len(data_ids):,} IDs")

        # Get EURASIA3 boundaries
        boundaries = {
            'EURASIA3': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': -180, 'X_MIN_END': -169}
        }
        bbox = boundaries['EURASIA3']

        # Filter vector file by bounding box
        in_bbox = gdf[
            (gdf.geometry.x >= bbox['X_MIN_START']) &
            (gdf.geometry.x <= bbox['X_MIN_END']) &
            (gdf.geometry.y >= bbox['Y_MIN_START']) &
            (gdf.geometry.y <= bbox['Y_MIN_END'])
            ]
        print(f"Lakes in EURASIA3 bounding box: {len(in_bbox):,}")

        # Check overlap with data IDs
        vector_ids_in_bbox = set(in_bbox['id_geohash'].values)
        overlap = data_ids.intersection(vector_ids_in_bbox)

        print(f"Data IDs overlapping with vector IDs in bbox: {len(overlap):,}")

        if len(overlap) == 0:
            print("\n⚠️ NO OVERLAP! Possible issues:")
            print("1. The data IDs don't exist in the vector file")
            print("2. The IDs exist but are outside the bounding box")
            print("3. The vector file uses different ID format")

            # Check if ANY data IDs exist in the vector file (regardless of bbox)
            all_vector_ids = set(gdf['id_geohash'].values)
            any_overlap = data_ids.intersection(all_vector_ids)
            print(f"\nData IDs that exist ANYWHERE in vector file: {len(any_overlap):,}")

            if len(any_overlap) > 0:
                print("✅ Data IDs DO exist in the vector file, but outside the bbox!")
                print("   The region boundaries might be wrong for EURASIA3")

                # Show where these IDs are located
                gdf_overlap = gdf[gdf['id_geohash'].isin(any_overlap)]
                print(f"\nLocations of overlapping EURASIA3 data IDs:")
                print(f"  Longitude: {gdf_overlap.geometry.x.min():.2f} to {gdf_overlap.geometry.x.max():.2f}")
                print(f"  Latitude: {gdf_overlap.geometry.y.min():.2f} to {gdf_overlap.geometry.y.max():.2f}")

                print(f"\nSuggested new boundaries for EURASIA3:")
                print(f"  X_MIN_START: {int(gdf_overlap.geometry.x.min())}")
                print(f"  X_MIN_END: {int(gdf_overlap.geometry.x.max()) + 1}")
                print(f"  Y_MIN_START: {int(gdf_overlap.geometry.y.min())}")
                print(f"  Y_MIN_END: {int(gdf_overlap.geometry.y.max()) + 1}")
            else:
                print("❌ Data IDs do NOT exist in the vector file AT ALL!")
                print("   The vector file and data files use different ID systems.")
                print(f"   Sample data ID: {next(iter(data_ids))}")
                print(f"   Sample vector ID: {next(iter(all_vector_ids))}")
        else:
            print(f"✅ Found {len(overlap):,} overlapping IDs!")
            print("   The data should be processable.")

        return {
            'data_ids': len(data_ids),
            'vector_ids_in_bbox': len(in_bbox),
            'overlap': len(overlap),
            'sample_data_ids': list(data_ids)[:5],
            'sample_vector_ids': list(vector_ids_in_bbox)[:5]
        }

    # Run this after the comparison
    result = debug_eurasia3_specific(env_path)
    logger.debug(result)
    logger.debug('RESULT FOR EURASIA3')

    logger.info("=" * 80)
    logger.info("COMPARING VECTOR FILES")
    logger.info("=" * 80)
    logger.info(f"Old vector file: {old_vector_file}")
    logger.info(f"New vector file: {new_vector_file}")

    # Run comparison for all regions
    regions_to_check = ['TEST', 'ALASKA', 'CANADA', 'EURASIA1', 'EURASIA2', 'EURASIA3']

    for region in regions_to_check:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"COMPARING FOR REGION: {region}")
        logger.info(f"{'=' * 60}")

        result = compare_vector_files(
            old_file_path=old_vector_file,
            new_file_path=new_vector_file,
            region=region,
            env_path=env_path
        )

        if result.get('error'):
            logger.error(f"Error comparing for {region}: {result['error']}")
        else:
            comparison = result.get('comparison', {})
            region_specific = comparison.get('region_specific', {})

            if region_specific:
                logger.info(f"  Old lakes in {region}: {region_specific.get('old_in_region', 0):,}")
                logger.info(f"  New lakes in {region}: {region_specific.get('new_in_region', 0):,}")
                logger.info(f"  Overlap in {region}: {region_specific.get('overlap_in_region', 0):,}")

                if region_specific.get('new_in_region', 0) > region_specific.get('old_in_region', 0):
                    logger.info(
                        f"  ✅ New file has {region_specific['new_in_region'] - region_specific['old_in_region']:,} more lakes")
                elif region_specific.get('new_in_region', 0) < region_specific.get('old_in_region', 0):
                    logger.info(
                        f"  ⚠️ Old file has {region_specific['old_in_region'] - region_specific['new_in_region']:,} more lakes")
                else:
                    logger.info(f"  Same number of lakes in both files")

    # ========== Get all regions ==========
    import utils.region_boundaries
    boundaries = utils.region_boundaries.get_region_boundaries()
    all_regions = list(boundaries.keys())
    logger.info(f"Available regions: {all_regions}")

    dynamic_world_data_dir = os.environ['dynamic_world_data']

    # ========== Determine if we should run ==========
    SHOULD_RUN = False
    summer_months = [6, 7, 8, 9]

    TODAY = datetime.now()
    TODAY_MONTH = TODAY.month

    if TODAY_MONTH - 1 in summer_months:
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 3:
            SHOULD_RUN = True
            logger.debug(f"Should run: {SHOULD_RUN}")

    if not SHOULD_RUN:
        logger.debug(f"Too early in the month to run downloads - exiting")
        return

    # ========== Prepare date to run ==========
    date_to_run = datetime(TODAY.year, TODAY_MONTH - 1, 1).strftime("%Y-%m")
    logger.info(f"Processing date: {date_to_run}")

    # ========== Process ALL regions (FAST - no verification) ==========
    results = {}
    success_count = 0
    failure_count = 0
    skipped_count = 0
    region_files = []

    for region in all_regions:
        result = process_region_fast(
            region=region,
            date_to_run=date_to_run,
            env_path=env_path,
            dynamic_world_data_dir=dynamic_world_data_dir
        )

        if result['success']:
            success_count += 1
            if 'merged_file' in result:
                region_files.append(result['merged_file'])
        else:
            failure_count += 1

        results[region] = result

    # ========== Summary ==========
    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total regions: {len(all_regions)}")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Failed: {failure_count}")
    logger.info(f"Skipped: {skipped_count}")

    # Log which regions succeeded/failed
    for region, result in results.items():
        status = "✅" if result['success'] else "❌"
        reason = result.get('reason', '')
        logger.info(f"  {status} {region}: {reason}")

    logger.info("Done!")


if __name__ == "__main__":

    main()