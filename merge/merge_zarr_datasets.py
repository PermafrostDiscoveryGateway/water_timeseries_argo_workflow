# batch_merge.py
import xarray as xr
from pathlib import Path
from water_timeseries.processing import DWDataset


def merge_all_zarrs(zarr_dir: str, output_file: str, batch_size: int = 5):
    """
    Merge multiple .zarr files into one.

    Args:
        zarr_dir: Directory containing .zarr files
        output_file: Output path for merged dataset
        batch_size: Number of files to merge at once (for memory efficiency)
    """
    zarr_files = sorted(Path(zarr_dir).glob("dynamic_world_split_*.zarr"))
    print(f"Found {len(zarr_files)} files to merge")

    # Function to load a dataset
    def load_ds(filepath):
        return DWDataset(xr.open_zarr(filepath), mask_data=False)

    # Merge in batches to avoid memory issues
    while len(zarr_files) > 1:
        batch = zarr_files[:batch_size]
        remaining = zarr_files[batch_size:]

        print(f"Merging batch of {len(batch)} files...")
        merged = load_ds(batch[0])
        for filepath in batch[1:]:
            merged = merged.merge(load_ds(filepath), how="id_geohash")

        # Save intermediate merged file
        temp_file = output_file.replace('.zarr', '_temp.zarr')
        merged.ds.to_zarr(temp_file, mode='w')

        # Prepare for next iteration
        zarr_files = [Path(temp_file)] + remaining

    # Final rename
    if zarr_files and zarr_files[0] != Path(output_file):
        Path(zarr_files[0]).rename(output_file)
        print(f"Saved merged dataset to {output_file}")

    print(f"Final dataset has {len(merged.ds.coords['id_geohash'])} lakes and {len(merged.ds.coords['date'])} dates")


# Usage
merge_all_zarrs(
    zarr_dir="/path/to/your/dynamic_world/zarr/files",
    output_file="/path/to/merged_dynamic_world.zarr",
    batch_size=5  # Adjust based on your memory
)