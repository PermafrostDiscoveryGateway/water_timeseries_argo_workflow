# split_parquet.py
import geopandas as gpd
import os
from pathlib import Path

# Load the full dataset
input_file = "/mnt/argo-filestore/water_timeseries/input/Nitze_etal_Lakes_filtered_full_set_V2d.parquet"
output_dir = "/mnt/argo-filestore/water_timeseries/input/split_lakes_1"

# Create output directory
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Load the data
print("Loading parquet file...")
gdf = gpd.read_parquet(input_file)
total_features = len(gdf)
print(f"Total features: {total_features}")

# Split into chunks of 5000 features each
chunk_size = 1000
num_chunks = (total_features + chunk_size - 1) // chunk_size

print(f"Splitting into {num_chunks} chunks of ~{chunk_size} features each")

for i in range(0, total_features, chunk_size):
    chunk_num = i // chunk_size + 1
    chunk = gdf.iloc[i:i+chunk_size]
    output_file = os.path.join(output_dir, f"lakes_chunk_{chunk_num:04d}.parquet")
    chunk.to_parquet(output_file)
    print(f"Saved {len(chunk)} features to {output_file}")

print(f"\nDone! Created {num_chunks} files in {output_dir}")