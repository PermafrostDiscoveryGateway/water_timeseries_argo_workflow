import os
import glob
import h5py
import netCDF4 as nc

split_dir = "/mnt/argo-filestore/water_timeseries/split_new_dynamic_world_data"
chunk_files = glob.glob(os.path.join(split_dir, "*.nc"))

corrupted_files = []
valid_files = []

for f in chunk_files:
    size = os.path.getsize(f)
    try:
        # Try to open with netCDF4
        with nc.Dataset(f, 'r') as ds:
            # Just check if we can read dimensions
            dims = list(ds.dimensions.keys())
            valid_files.append((f, size, dims))
    except Exception as e:
        corrupted_files.append((f, size, str(e)[:100]))

print(f"Total files: {len(chunk_files)}")
print(f"Valid: {len(valid_files)}")
print(f"Corrupted: {len(corrupted_files)}")

if corrupted_files:
    print("\nFirst 5 corrupted files:")
    for f, size, error in corrupted_files[:5]:
        print(f"  {os.path.basename(f)}: {size} bytes - {error}")