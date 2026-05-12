# split_parquet_two_step.py
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path

input_file = "/mnt/argo-filestore/water_timeseries/input/Nitze_etal_Lakes_filtered_full_set_V2d.parquet"
output_dir = "/mnt/argo-filestore/water_timeseries/input/split_lakes_2"

Path(output_dir).mkdir(parents=True, exist_ok=True)

# Step 1: Get total rows without loading the whole file
import pyarrow.parquet as pq

parquet_file = pq.ParquetFile(input_file)
total_rows = parquet_file.metadata.num_rows
print(f"Total features: {total_rows}")

# Step 2: Read and split in chunks
chunk_size = 500
chunk_num = 0

print(f"Splitting into chunks of {chunk_size} features...")

for chunk in pd.read_parquet(input_file, chunksize=chunk_size):
    chunk_num += 1
    output_file = os.path.join(output_dir, f"lakes_chunk_{chunk_num:04d}.parquet")

    # Convert to geopandas if your downloader needs geometry
    # gdf_chunk = gpd.GeoDataFrame(chunk, geometry='geometry')
    # gdf_chunk.to_parquet(output_file)

    # Or just save as pandas (geometry column will be preserved as WKB)
    chunk.to_parquet(output_file, index=False)

    print(f"Saved {len(chunk)} features to {output_file}")

print(f"\n✅ Done! Created {chunk_num} files")