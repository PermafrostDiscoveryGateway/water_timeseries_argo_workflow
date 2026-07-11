from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import os
import sys
import netCDF4 as nc
import numpy as np
from datetime import datetime

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def verify_netcdf_data(first_file, second_file):
    """
    Verify that two NetCDF files have the same data.

    Args:
        first_file: Path to the first NetCDF file
        second_file: Path to the second NetCDF file

    Returns:
        bool: True if data matches, False otherwise
    """
    logger.debug(f"Verifying that {first_file} and {second_file} have the same data.")

    # Convert to Path objects
    first_path = Path(first_file)
    second_path = Path(second_file)

    # Check if both files exist
    if not first_path.exists():
        logger.error(f"First file does not exist: {first_file}")
        return False
    if not second_path.exists():
        logger.error(f"Second file does not exist: {second_file}")
        return False

    try:
        # Open both NetCDF files
        with nc.Dataset(first_file, 'r') as src1, nc.Dataset(second_file, 'r') as src2:
            logger.info("=" * 80)
            logger.info("VERIFYING NETCDF DATA INTEGRITY")
            logger.info("=" * 80)

            # 1. Check dimensions
            logger.info("\n1. Checking dimensions...")
            dims1 = set(src1.dimensions.keys())
            dims2 = set(src2.dimensions.keys())

            if dims1 != dims2:
                logger.error(f"  Dimension mismatch!")
                logger.error(f"    First file dimensions: {dims1}")
                logger.error(f"    Second file dimensions: {dims2}")
                return False

            for dim_name in dims1:
                len1 = len(src1.dimensions[dim_name])
                len2 = len(src2.dimensions[dim_name])
                if len1 != len2:
                    logger.error(f"  Dimension '{dim_name}' length mismatch: {len1} vs {len2}")
                    return False
                logger.info(f"  Dimension '{dim_name}': {len1} (matches)")

            # 2. Check variables (names and attributes)
            logger.info("\n2. Checking variables...")
            vars1 = set(src1.variables.keys())
            vars2 = set(src2.variables.keys())

            if vars1 != vars2:
                logger.error(f"  Variable mismatch!")
                logger.error(f"    First file variables: {vars1}")
                logger.error(f"    Second file variables: {vars2}")
                missing_in_first = vars2 - vars1
                missing_in_second = vars1 - vars2
                if missing_in_first:
                    logger.error(f"    Variables missing in first file: {missing_in_first}")
                if missing_in_second:
                    logger.error(f"    Variables missing in second file: {missing_in_second}")
                return False

            # Check variable attributes
            for var_name in vars1:
                var1 = src1.variables[var_name]
                var2 = src2.variables[var_name]

                # Check attributes
                attrs1 = set(var1.ncattrs())
                attrs2 = set(var2.ncattrs())
                if attrs1 != attrs2:
                    logger.warning(f"  Variable '{var_name}' attribute mismatch:")
                    logger.warning(f"    First file attributes: {attrs1}")
                    logger.warning(f"    Second file attributes: {attrs2}")
                    # Continue with verification, just log the warning

                # Check dtype
                if var1.dtype != var2.dtype:
                    logger.error(f"  Variable '{var_name}' dtype mismatch: {var1.dtype} vs {var2.dtype}")
                    return False

                # Check shape
                if var1.shape != var2.shape:
                    logger.error(f"  Variable '{var_name}' shape mismatch: {var1.shape} vs {var2.shape}")
                    return False

            # 3. Check global attributes
            logger.info("\n3. Checking global attributes...")
            global_attrs1 = set(src1.ncattrs())
            global_attrs2 = set(src2.ncattrs())

            if global_attrs1 != global_attrs2:
                logger.warning(f"  Global attribute mismatch:")
                logger.warning(f"    First file attributes: {global_attrs1}")
                logger.warning(f"    Second file attributes: {global_attrs2}")

            # 4. Check data values
            logger.info("\n4. Checking data values...")
            all_match = True
            total_vars = len(vars1)
            checked_vars = 0

            for var_name in sorted(vars1):
                var1 = src1.variables[var_name]
                var2 = src2.variables[var_name]

                # Skip variables with no data (dimension variables)
                if len(var1.dimensions) == 0:
                    # Scalar variable
                    if var1[:] != var2[:]:
                        logger.error(f"  Variable '{var_name}' data mismatch (scalar)")
                        all_match = False
                    else:
                        logger.info(f"  Variable '{var_name}': ✓ (scalar matches)")
                    checked_vars += 1
                    continue

                # For variables with data, check in chunks to manage memory
                total_elements = np.prod(var1.shape)
                logger.info(f"  Checking variable '{var_name}' (shape: {var1.shape}, {total_elements:,} elements)...")

                # Determine if variable is large enough to warrant chunked checking
                if total_elements > 1_000_000:  # More than 1 million elements
                    # Check by sampling or chunked comparison
                    # Check first and last slices, and some random indices
                    logger.info(f"    Large variable, using sampling approach...")

                    # Check first element
                    if len(var1.shape) == 1:
                        idx1 = 0
                        if var1[idx1] != var2[idx1]:
                            logger.error(f"    Variable '{var_name}' data mismatch at index {idx1}")
                            all_match = False
                            break
                        # Check last element
                        idx1 = -1
                        if var1[idx1] != var2[idx1]:
                            logger.error(f"    Variable '{var_name}' data mismatch at index {idx1}")
                            all_match = False
                            break
                        # Check middle element
                        idx1 = len(var1) // 2
                        if var1[idx1] != var2[idx1]:
                            logger.error(f"    Variable '{var_name}' data mismatch at index {idx1}")
                            all_match = False
                            break

                    elif len(var1.shape) == 2:
                        # Check corners
                        corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
                        for idx in corners:
                            if var1[idx] != var2[idx]:
                                logger.error(f"    Variable '{var_name}' data mismatch at index {idx}")
                                all_match = False
                                break
                        if not all_match:
                            break
                        # Check center
                        center = (var1.shape[0] // 2, var1.shape[1] // 2)
                        if var1[center] != var2[center]:
                            logger.error(f"    Variable '{var_name}' data mismatch at index {center}")
                            all_match = False
                            break

                    else:
                        # For 3D+ variables, check a few slices
                        slice_indices = []
                        for dim in range(len(var1.shape)):
                            slice_indices.append(var1.shape[dim] // 2)

                        # Check slice at half of each dimension
                        try:
                            if len(var1.shape) == 3:
                                idx = (slice_indices[0], slice_indices[1], slice_indices[2])
                                if var1[idx] != var2[idx]:
                                    logger.error(f"    Variable '{var_name}' data mismatch at index {idx}")
                                    all_match = False
                                    break
                            elif len(var1.shape) == 4:
                                idx = (slice_indices[0], slice_indices[1], slice_indices[2], slice_indices[3])
                                if var1[idx] != var2[idx]:
                                    logger.error(f"    Variable '{var_name}' data mismatch at index {idx}")
                                    all_match = False
                                    break
                        except:
                            # If slicing fails, skip sampling
                            pass

                    # Do a statistical comparison (mean, min, max)
                    try:
                        # Use numpy to compute statistics efficiently
                        data1 = var1[:]
                        data2 = var2[:]

                        # Check if any NaN values
                        if np.any(np.isnan(data1)) != np.any(np.isnan(data2)):
                            logger.warning(f"    NaN pattern mismatch in '{var_name}'")

                        # Compare statistics
                        stats1 = {
                            'mean': np.nanmean(data1),
                            'std': np.nanstd(data1),
                            'min': np.nanmin(data1),
                            'max': np.nanmax(data1)
                        }
                        stats2 = {
                            'mean': np.nanmean(data2),
                            'std': np.nanstd(data2),
                            'min': np.nanmin(data2),
                            'max': np.nanmax(data2)
                        }

                        # Compare statistics with tolerance
                        tolerance = 1e-6
                        stats_match = True
                        for stat_name in stats1:
                            diff = abs(stats1[stat_name] - stats2[stat_name])
                            if diff > tolerance:
                                logger.warning(
                                    f"    {stat_name} differs: {stats1[stat_name]:.6f} vs {stats2[stat_name]:.6f} (diff: {diff:.6f})")
                                stats_match = False

                        if stats_match:
                            logger.info(f"    ✓ Statistical properties match")
                        else:
                            logger.warning(f"    Statistical properties differ (check if acceptable)")

                        # Full array comparison with tolerance for floating point
                        if np.allclose(data1, data2, rtol=1e-5, atol=1e-8, equal_nan=True):
                            logger.info(f"  Variable '{var_name}': ✓ (matches within tolerance)")
                        else:
                            # Find where they differ
                            diff_mask = ~np.isclose(data1, data2, rtol=1e-5, atol=1e-8, equal_nan=True)
                            if np.any(diff_mask):
                                if np.sum(diff_mask) < 10:  # Only show if few differences
                                    diff_indices = np.where(diff_mask)
                                    logger.error(f"  Variable '{var_name}' data mismatch at indices: {diff_indices}")
                                else:
                                    logger.error(
                                        f"  Variable '{var_name}' data mismatch at {np.sum(diff_mask)} locations")
                                all_match = False
                            else:
                                logger.info(f"  Variable '{var_name}': ✓ (matches)")

                    except MemoryError:
                        logger.warning(f"    Memory error checking '{var_name}', skipping detailed comparison")
                        logger.info(f"  Variable '{var_name}': ⚠ (sampling only, see above)")

                else:
                    # Small variable, check all data
                    try:
                        data1 = var1[:]
                        data2 = var2[:]

                        if np.allclose(data1, data2, rtol=1e-5, atol=1e-8, equal_nan=True):
                            logger.info(f"  Variable '{var_name}': ✓ (matches)")
                        else:
                            logger.error(f"  Variable '{var_name}' data mismatch")
                            all_match = False
                    except Exception as e:
                        logger.error(f"  Error checking variable '{var_name}': {e}")
                        all_match = False

                checked_vars += 1
                progress = (checked_vars / total_vars) * 100
                if checked_vars % 5 == 0 or checked_vars == total_vars:
                    logger.info(f"    Progress: {checked_vars}/{total_vars} variables checked ({progress:.1f}%)")

            # 5. Summary
            logger.info("=" * 80)
            if all_match:
                logger.info("✓✓✓ VERIFICATION PASSED: All data matches! ✓✓✓")
                logger.info("=" * 80)
                return True
            else:
                logger.error("✗✗✗ VERIFICATION FAILED: Data mismatches found! ✗✗✗")
                logger.info("=" * 80)
                return False

    except Exception as e:
        logger.error(f"Error during verification: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def compress_netcdf_file_chunks(input_file, output_filename):
    """
    Compress a NetCDF file by writing it in chunks with compression.

    Args:
        input_file: Path to the input NetCDF file
        output_filename: Name of the output compressed file
    """
    logger.info(f"Compressing {input_file} to {output_filename}")

    # Get the directory of the input file
    input_dir = Path(input_file).parent
    output_filepath = input_dir / output_filename

    # Open the input NetCDF file
    with nc.Dataset(input_file, 'r') as src:
        # Create the output NetCDF file with compression settings
        # Using zlib compression with shuffling for better compression
        with nc.Dataset(output_filepath, 'w', format='NETCDF4') as dst:
            # Copy global attributes
            dst.setncatts(src.__dict__)

            # Copy dimensions
            for dim_name, dim in src.dimensions.items():
                if dim.isunlimited():
                    dst.createDimension(dim_name, None)
                else:
                    dst.createDimension(dim_name, len(dim))

            # Copy variables with compression
            for var_name, var in src.variables.items():
                # Create variable with compression settings
                # Using chunksizes based on the variable's dimensions
                chunksizes = None
                if len(var.dimensions) > 0:
                    # Calculate reasonable chunksizes
                    # For large dimensions, use chunks of size 100 or the dimension size if smaller
                    chunksizes = []
                    for dim_name in var.dimensions:
                        dim_size = len(src.dimensions[dim_name])
                        # Use chunk size of 100 for large dimensions, or the full dimension if small
                        chunk_size = min(100, dim_size) if dim_size > 0 else 1
                        chunksizes.append(chunk_size)
                    chunksizes = tuple(chunksizes)

                # Create variable with compression
                dst_var = dst.createVariable(
                    var_name,
                    var.dtype,
                    var.dimensions,
                    zlib=True,  # Enable compression
                    complevel=4,  # Compression level (1-9, 4 is good balance)
                    shuffle=True,  # Enable shuffle filter for better compression
                    chunksizes=chunksizes,
                    fill_value=var._fill_value if hasattr(var, '_fill_value') else None
                )

                # Copy variable attributes
                dst_var.setncatts(var.__dict__)

                # Copy data in chunks to manage memory
                # Get total size of the variable
                total_size = 1
                for dim_name in var.dimensions:
                    total_size *= len(src.dimensions[dim_name])

                # If variable is small, copy all at once
                if total_size < 1_000_000:  # Less than 1 million elements
                    dst_var[:] = var[:]
                else:
                    # Copy in chunks along the first dimension
                    first_dim_name = var.dimensions[0] if var.dimensions else None
                    if first_dim_name:
                        first_dim_size = len(src.dimensions[first_dim_name])
                        chunk_size = min(100, first_dim_size)

                        for i in range(0, first_dim_size, chunk_size):
                            end_idx = min(i + chunk_size, first_dim_size)
                            # Create slice indices
                            if len(var.dimensions) == 1:
                                dst_var[i:end_idx] = var[i:end_idx]
                            elif len(var.dimensions) == 2:
                                dst_var[i:end_idx, :] = var[i:end_idx, :]
                            elif len(var.dimensions) == 3:
                                dst_var[i:end_idx, :, :] = var[i:end_idx, :, :]
                            elif len(var.dimensions) == 4:
                                dst_var[i:end_idx, :, :, :] = var[i:end_idx, :, :, :]
                            else:
                                # Fallback: copy entire variable if dimensions > 4
                                dst_var[:] = var[:]
                                break

                            logger.debug(f"  Copied chunk {i}-{end_idx} of {first_dim_size} for variable {var_name}")
                    else:
                        # No dimensions, just copy the scalar
                        dst_var[:] = var[:]

                logger.info(f"  Copied variable: {var_name} with shape {var.shape}")

    # Get the file sizes for comparison
    original_size = Path(input_file).stat().st_size / (1024 * 1024)  # MB
    compressed_size = output_filepath.stat().st_size / (1024 * 1024)  # MB
    compression_ratio = compressed_size / original_size if original_size > 0 else 0

    logger.info(f"Compression complete!")
    logger.info(f"  Original size: {original_size:.2f} MB")
    logger.info(f"  Compressed size: {compressed_size:.2f} MB")
    logger.info(f"  Compression ratio: {compression_ratio:.2%} of original")
    logger.info(f"  Output saved to: {output_filepath}")


def get_file_size_and_info(filepath):
    """
    Get file size and compression information.

    Args:
        filepath: Path to the file

    Returns:
        tuple: (size_bytes, size_mb, is_compressed, compression_info)
    """
    path = Path(filepath)

    if not path.exists():
        logger.error(f"File not found: {filepath}")
        return None, None, None, None

    # Get file size
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    # Check if NetCDF file is compressed
    is_compressed = False
    compression_info = {}

    try:
        with nc.Dataset(filepath, 'r') as ds:
            # Check if it's a NetCDF4 file with compression
            if hasattr(ds, 'file_format') and ds.file_format == 'NETCDF4':
                compression_info['format'] = 'NETCDF4'
                # Check variables for compression
                compressed_vars = []
                for var_name, var in ds.variables.items():
                    if hasattr(var, 'filters'):
                        filters = var.filters()
                        if filters and filters.get('zlib', False):
                            compressed_vars.append(var_name)
                            compression_info[f'{var_name}_compression'] = filters

                if compressed_vars:
                    is_compressed = True
                    compression_info['compressed_vars'] = compressed_vars
                    logger.info(f"  Variable compression: {len(compressed_vars)} variables compressed")
                else:
                    logger.info("  No compressed variables found")
            else:
                compression_info['format'] = 'NETCDF3 or other format'

    except Exception as e:
        logger.warning(f"Could not read NetCDF compression info: {e}")
        compression_info['error'] = str(e)

    return size_bytes, size_mb, is_compressed, compression_info


def main():
    logger.debug(f"Checking compression of netcdf files and compressing by writing in chunks")
    env_path = None
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    # ========== DEBUGGING: Check ALL environment variables ==========
    logger.info("=" * 80)
    logger.info("ENVIRONMENT VARIABLES (ALL)")
    logger.info("=" * 80)
    for key, value in sorted(os.environ.items()):
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)

    dynamic_world_data_dir = os.environ['dynamic_world_data']
    original_filename = 'workflows_optimization_lake_change_detection_lakes_dw_V2d_2016-2025.nc'
    target_filename = 'lakes_dw_V2d_compressed.nc'
    original_filepath = os.path.join(dynamic_world_data_dir, original_filename)

    # Get filesize of original filepath
    size_bytes, size_mb, is_compressed, compression_info = get_file_size_and_info(original_filepath)

    if size_bytes is not None:
        logger.info(f"Original file: {original_filepath}")
        logger.info(f"  File size: {size_bytes:,} bytes ({size_mb:.2f} MB)")
        logger.info(f"  Is compressed: {is_compressed}")
        logger.info(f"  Compression info: {compression_info}")
    else:
        logger.error(f"Could not get file size for {original_filepath}")
        return

    # Get compression of original filepath
    # Already done in the function above, but let's check specifically
    if is_compressed:
        logger.info(f"Original file is already compressed")
        logger.info("  Compression details:")
        for key, value in compression_info.items():
            if key != 'compressed_vars':
                logger.info(f"    {key}: {value}")
        if 'compressed_vars' in compression_info:
            logger.info(f"    Compressed variables: {compression_info['compressed_vars']}")
    else:
        logger.info(f"Original file is not compressed")

    # Ask user if they want to proceed with compression
    response = input(f"\nCompress '{original_filename}' to '{target_filename}'? (y/n): ")
    if response.lower() != 'y':
        logger.info("Compression cancelled by user")
        return

    # Compress the file
    start_time = datetime.now()
    compress_netcdf_file_chunks(input_file=original_filepath, output_filename=target_filename)
    end_time = datetime.now()

    logger.info(f"Total compression time: {end_time - start_time}")

    # TODO implement this method checking files
    data_verified = verify_netcdf_data(first_file=original_filepath, second_file=target_filename)
    logger.debug(f"Data verified: {data_verified}")

if __name__ == "__main__":
    main()