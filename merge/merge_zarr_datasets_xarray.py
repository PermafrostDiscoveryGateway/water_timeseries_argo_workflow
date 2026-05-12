# simple_merge.py
import xarray as xr
from pathlib import Path

zarr_dir = "/path/to/your/chunks"
zarr_files = sorted(Path(zarr_dir).glob("dynamic_world_split_*.zarr"))

# Concatenate along the id_geohash dimension
merged = xr.concat([xr.open_zarr(f) for f in zarr_files], dim="id_geohash")

# Save
merged.to_zarr("merged_dynamic_world.zarr", mode='w')
print(f"Merged {len(zarr_files)} files into dataset with {len(merged.id_geohash)} lakes")