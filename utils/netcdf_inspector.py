import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger


def inspect_netcdf(file_path: str, verbose: bool = True) -> dict:
    """
    Inspect a NetCDF file and print out its structure, dimensions, variables, and statistics.

    Args:
        file_path: Path to the NetCDF file
        verbose: If True, prints detailed information

    Returns:
        dict: Dictionary containing the inspection results
    """

    def print_section(title, char='='):
        """Print a section header."""
        if verbose:
            print(f"\n{char * 80}")
            print(f"{title}")
            print(f"{char * 80}")

    def get_file_size_gb(path):
        """Get file size in GB."""
        if Path(path).exists():
            return Path(path).stat().st_size / (1024 ** 3)
        return 0

    # Open the dataset
    try:
        ds = xr.open_dataset(file_path)
    except Exception as e:
        logger.error(f"Error opening file: {e}")
        return {'error': str(e)}

    result = {
        'file_path': str(file_path),
        'file_size_gb': get_file_size_gb(file_path),
        'dimensions': {},
        'variables': {},
        'coordinates': {},
        'attributes': {},
        'data_stats': {}
    }

    # ========== BASIC FILE INFO ==========
    if verbose:
        print_section("BASIC FILE INFORMATION")
        print(f"File: {file_path}")
        print(f"Size: {result['file_size_gb']:.4f} GB")
        print(f"Dimensions: {list(ds.dims)}")
        print(f"Data Variables: {len(ds.data_vars)}")
        print(f"Coordinates: {len(ds.coords)}")

    # ========== DIMENSIONS ==========
    print_section("DIMENSIONS")
    for dim_name, dim_size in ds.dims.items():
        print(f"  {dim_name}: {dim_size:,}")
        result['dimensions'][dim_name] = dim_size

    # ========== COORDINATES ==========
    print_section("COORDINATES")
    for coord_name in ds.coords:
        coord = ds[coord_name]
        print(f"  {coord_name}:")
        print(f"    dtype: {coord.dtype}")
        print(f"    shape: {coord.shape}")

        # Show first few values
        if len(coord) > 0:
            sample_size = min(5, len(coord))
            values = coord.values[:sample_size]
            print(f"    sample: {values}")

        result['coordinates'][coord_name] = {
            'dtype': str(coord.dtype),
            'shape': coord.shape,
            'ndim': coord.ndim
        }

    # ========== DATA VARIABLES ==========
    print_section("DATA VARIABLES")

    for var_name in ds.data_vars:
        var = ds[var_name]

        print(f"\n  Variable: {var_name}")
        print(f"    dtype: {var.dtype}")
        print(f"    shape: {var.shape}")
        print(f"    ndim: {var.ndim}")
        print(f"    dimensions: {list(var.dims)}")

        # Show first few values
        try:
            if var.size > 0:
                # For large arrays, show a sample
                flat_values = var.values.flatten()
                sample_size = min(5, len(flat_values))

                if np.issubdtype(var.dtype, np.number):
                    # Numeric data - show stats
                    valid_values = flat_values[~np.isnan(flat_values)] if np.issubdtype(var.dtype,
                                                                                        np.floating) else flat_values
                    if len(valid_values) > 0:
                        print(f"    sample: {valid_values[:sample_size]}")
                        print(f"    min: {np.min(valid_values):.4f}" if np.issubdtype(var.dtype, np.number) else "")
                        print(f"    max: {np.max(valid_values):.4f}" if np.issubdtype(var.dtype, np.number) else "")
                        print(f"    mean: {np.mean(valid_values):.4f}" if np.issubdtype(var.dtype, np.number) else "")
                        print(f"    std: {np.std(valid_values):.4f}" if np.issubdtype(var.dtype, np.number) else "")
                        print(
                            f"    non-NaN count: {len(valid_values):,}/{len(flat_values):,}" if np.issubdtype(var.dtype,
                                                                                                              np.floating) else f"    count: {len(flat_values):,}")
                else:
                    # Non-numeric data - show sample
                    print(f"    sample: {flat_values[:sample_size]}")

                # Show unique values for categorical data
                if var.dtype == 'object' or var.dtype == 'str':
                    unique_values = np.unique(flat_values)
                    if len(unique_values) <= 20:
                        print(f"    unique values: {list(unique_values)}")
                    else:
                        print(f"    unique values: {len(unique_values):,} (showing first 5)")
                        print(f"    sample unique: {list(unique_values[:5])}")
        except Exception as e:
            print(f"    Could not inspect values: {e}")

        # Check encoding/compression
        if var.encoding:
            has_compression = var.encoding.get('zlib', False) or var.encoding.get('complevel', 0) > 0
            print(f"    compression: {'Yes' if has_compression else 'No'}")
            if has_compression:
                print(f"    compression level: {var.encoding.get('complevel', 'N/A')}")
                print(f"    shuffle: {var.encoding.get('shuffle', 'N/A')}")

        result['variables'][var_name] = {
            'dtype': str(var.dtype),
            'shape': var.shape,
            'ndim': var.ndim,
            'dims': list(var.dims),
            'size': var.size,
            'encoding': var.encoding if var.encoding else None
        }

        # Store stats for numeric variables
        if np.issubdtype(var.dtype, np.number) and var.size > 0:
            try:
                flat_values = var.values.flatten()
                valid_values = flat_values[~np.isnan(flat_values)] if np.issubdtype(var.dtype,
                                                                                    np.floating) else flat_values
                if len(valid_values) > 0:
                    result['data_stats'][var_name] = {
                        'min': float(np.min(valid_values)),
                        'max': float(np.max(valid_values)),
                        'mean': float(np.mean(valid_values)),
                        'std': float(np.std(valid_values)),
                        'count': len(valid_values),
                        'total': len(flat_values),
                        'null_count': len(flat_values) - len(valid_values) if np.issubdtype(var.dtype,
                                                                                            np.floating) else 0
                    }
            except Exception as e:
                pass

    # ========== ATTRIBUTES ==========
    if ds.attrs:
        print_section("GLOBAL ATTRIBUTES")
        for attr_name, attr_value in ds.attrs.items():
            print(f"  {attr_name}: {attr_value}")
            result['attributes'][attr_name] = attr_value
    else:
        print_section("GLOBAL ATTRIBUTES")
        print("  No global attributes found")

    # ========== SUMMARY ==========
    print_section("SUMMARY")
    print(f"File: {Path(file_path).name}")
    print(f"Size: {result['file_size_gb']:.4f} GB")
    print(f"Dimensions:")
    for dim_name, dim_size in ds.dims.items():
        print(f"  {dim_name}: {dim_size:,}")
    print(f"Variables: {len(ds.data_vars)}")
    print(f"Coordinates: {len(ds.coords)}")
    print(f"Global attributes: {len(ds.attrs)}")

    # ========== DATA TYPES SUMMARY ==========
    print_section("DATA TYPES SUMMARY")
    dtype_counts = {}
    for var_name, var_info in result['variables'].items():
        dtype = var_info['dtype']
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} variables")

    # ========== CHECK FOR MISSING DATA ==========
    print_section("MISSING DATA CHECK")
    for var_name, var in ds.data_vars.items():
        if np.issubdtype(var.dtype, np.floating):
            if var.size > 0:
                nan_count = np.isnan(var.values).sum()
                total = var.size
                if nan_count > 0:
                    print(f"  {var_name}: {nan_count:,}/{total:,} NaN values ({nan_count / total * 100:.2f}%)")
                else:
                    print(f"  {var_name}: No NaN values")
        elif var.dtype == 'object':
            null_count = pd.isnull(var.values).sum()
            if null_count > 0:
                print(f"  {var_name}: {null_count:,} null values")

    # Close the dataset
    ds.close()

    result['total_variables'] = len(ds.data_vars)
    result['total_coordinates'] = len(ds.coords)
    result['total_attributes'] = len(ds.attrs)
    result['dimension_info'] = dict(ds.dims)

    print_section("INSPECTION COMPLETE", '=')

    return result


# =============================================================================
# SIMPLE USAGE
# =============================================================================

def quick_look(file_path: str):
    """
    Quick and simple summary of a NetCDF file.

    Args:
        file_path: Path to the NetCDF file
    """
    print("\n" + "=" * 60)
    print(f"QUICK LOOK: {Path(file_path).name}")
    print("=" * 60)

    try:
        ds = xr.open_dataset(file_path)

        # Basic info
        print(f"\n📁 File size: {Path(file_path).stat().st_size / (1024 ** 2):.2f} MB")
        print(f"📊 Dimensions:")
        for dim, size in ds.dims.items():
            print(f"   {dim}: {size:,}")

        print(f"\n📋 Variables ({len(ds.data_vars)}):")
        for var in ds.data_vars:
            dtype = ds[var].dtype
            shape = ds[var].shape
            print(f"   {var}: {dtype}, shape={shape}")

        print(f"\n📍 Coordinates ({len(ds.coords)}):")
        for coord in ds.coords:
            print(f"   {coord}")

        print("\n" + "=" * 60)

        ds.close()

    except Exception as e:
        print(f"❌ Error: {e}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import sys

    # Use command line argument or prompt
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter path to NetCDF file: ")

    # Full inspection
    inspect_netcdf(file_path, verbose=True)

    # Or just quick look
    # quick_look(file_path)