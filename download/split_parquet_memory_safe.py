# split_parquet_dask.py
import dask.dataframe as dd
import os
from pathlib import Path

input_file = "/mnt/argo-filestore/water_timeseries/input/Nitze_etal_Lakes_filtered_full_set_V2d.parquet"
output_dir = "/mnt/argo-filestore/water_timeseries/input/split_lakes_1"

Path(output_dir).mkdir(parents=True, exist_ok=True)

# Read with dask (lazy loading - doesn't load into memory)
print("Opening parquet file with dask...")
ddf = dd.read_parquet(input_file)

# Get total rows (requires compute, but only metadata)
total_rows = len(ddf)
print(f"Total features: {total_rows}")

# Repartition into chunks of roughly 500 rows
chunk_size = 500
n_partitions = max(1, total_rows // chunk_size)
ddf = ddf.repartition(npartitions=n_partitions)

# Write each partition as a separate file
print(f"Writing {n_partitions} chunks...")
for i, partition in enumerate(ddf.partitions):
    output_file = os.path.join(output_dir, f"lakes_chunk_{i+1:04d}.parquet")
    partition.to_parquet(output_file, index=False)
    print(f"Saved partition {i+1} to {output_file}")

print(f"\n✅ Done! Created {n_partitions} files")