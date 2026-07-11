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


if __name__ == "__main__":
    main()