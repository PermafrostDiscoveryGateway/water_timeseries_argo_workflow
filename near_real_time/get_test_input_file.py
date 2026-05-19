import xarray as xr
import numpy as np
from pathlib import Path
import random


def extract_lake_subset(
        input_nc_file: str | Path,
        output_nc_file: str | Path,
        num_lakes: int = 5000,
        selection_method: str = "random",  # "random", "first", "last"
        random_seed: int = 42
) -> Path:
    """
    Extract a subset of lakes from a large Dynamic World NetCDF file.

    Parameters
    ----------
    input_nc_file : str or Path
        Path to the original large NetCDF file
    output_nc_file : str or Path
        Path where the subset NetCDF will be saved
    num_lakes : int
        Number of lakes to extract (default: 5000)
    selection_method : str
        Method to select lakes: "random", "first", "last"
    random_seed : int
        Random seed for reproducibility (default: 42)

    Returns
    -------
    Path
        Path to the created subset file
    """
    input_nc_file = Path(input_nc_file)
    output_nc_file = Path(output_nc_file)

    print(f"Opening {input_nc_file}...")
    ds = xr.open_dataset(input_nc_file)

    all_lake_ids = ds.id_geohash.values
    total_lakes = len(all_lake_ids)
    print(f"Total lakes in original file: {total_lakes}")

    # Select lakes based on method
    if selection_method == "random":
        random.seed(random_seed)
        selected_indices = random.sample(range(total_lakes), min(num_lakes, total_lakes))
        selected_indices = sorted(selected_indices)  # Keep original order
    elif selection_method == "first":
        selected_indices = range(min(num_lakes, total_lakes))
    elif selection_method == "last":
        selected_indices = range(max(0, total_lakes - num_lakes), total_lakes)
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")

    selected_lake_ids = all_lake_ids[selected_indices]
    print(f"Selected {len(selected_lake_ids)} lakes using '{selection_method}' method")

    # Extract subset
    print("Extracting subset of data...")
    ds_subset = ds.sel(id_geohash=selected_lake_ids)

    # Save to new NetCDF file
    print(f"Saving to {output_nc_file}...")
    ds_subset.to_netcdf(output_nc_file)

    # Get file sizes
    original_size_gb = input_nc_file.stat().st_size / (1024 ** 3)
    subset_size_gb = output_nc_file.stat().st_size / (1024 ** 3)

    print(f"\n✅ Successfully created subset file:")
    print(f"  - Original file: {original_size_gb:.2f} GB")
    print(f"  - Subset file: {subset_size_gb:.2f} GB")
    print(f"  - Reduction: {(1 - subset_size_gb / original_size_gb) * 100:.1f}%")
    print(f"  - Lakes: {len(selected_lake_ids)} / {total_lakes}")
    print(f"  - Output path: {output_nc_file}")

    # Close the dataset
    ds.close()
    ds_subset.close()

    return output_nc_file


def extract_lake_subset_with_specific_ids(
        input_nc_file: str | Path,
        output_nc_file: str | Path,
        lake_ids: list
) -> Path:
    """
    Extract specific lake IDs from the NetCDF file.

    Parameters
    ----------
    input_nc_file : str or Path
        Path to the original large NetCDF file
    output_nc_file : str or Path
        Path where the subset NetCDF will be saved
    lake_ids : list
        List of specific lake IDs to extract

    Returns
    -------
    Path
        Path to the created subset file
    """
    input_nc_file = Path(input_nc_file)
    output_nc_file = Path(output_nc_file)

    print(f"Opening {input_nc_file}...")
    ds = xr.open_dataset(input_nc_file)

    # Find which IDs exist in the dataset
    all_lake_ids = set(ds.id_geohash.values)
    valid_ids = [lid for lid in lake_ids if lid in all_lake_ids]

    if not valid_ids:
        raise ValueError("None of the provided lake IDs were found in the dataset")

    print(f"Found {len(valid_ids)} out of {len(lake_ids)} requested lake IDs")

    # Extract subset
    print("Extracting subset of data...")
    ds_subset = ds.sel(id_geohash=valid_ids)

    # Save to new NetCDF file
    print(f"Saving to {output_nc_file}...")
    ds_subset.to_netcdf(output_nc_file)

    # Get file size
    subset_size_mb = output_nc_file.stat().st_size / (1024 ** 2)

    print(f"\n✅ Successfully created subset file:")
    print(f"  - Subset file size: {subset_size_mb:.2f} MB")
    print(f"  - Lakes extracted: {len(valid_ids)}")
    print(f"  - Output path: {output_nc_file}")

    ds.close()
    ds_subset.close()

    return output_nc_file


# Example usage in your main script
if __name__ == "__main__":
    # For testing: create a small subset of your large file
    large_file = Path("/data/water_timeseries/dynamic_world_data/lakes_dw_V2d.nc")
    test_file = Path("/data/water_timeseries/dynamic_world_data/lakes_dw_V2d_test_5000.nc")

    # Extract 5000 random lakes for testing
    extract_lake_subset(
        input_nc_file=large_file,
        output_nc_file=test_file,
        num_lakes=5000,
        selection_method="random"
    )

    # Now run your NRT analysis on the test file
    # precompute_nrt_breakpoints(
    #     input_nc_file=test_file,
    #     output_dir="/data/water_timeseries/test_output",
    #     lake_chunk_size=500
    # )