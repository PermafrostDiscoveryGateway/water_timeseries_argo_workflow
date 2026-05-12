# split_parquet_simple_fix.py
import pyarrow.parquet as pq
import pandas as pd
import os
from pathlib import Path

input_file = "/mnt/argo-filestore/water_timeseries/input/Nitze_etal_Lakes_filtered_full_set_V2d.parquet"
output_dir = "/mnt/argo-filestore/water_timeseries/input/split_lakes_1"

Path(output_dir).mkdir(parents=True, exist_ok=True)

# Open the parquet file
parquet_file = pq.ParquetFile(input_file)
total_rows = parquet_file.metadata.num_rows
print(f"Total features: {total_rows}")

# Get the schema to use for all chunks
schema = parquet_file.schema

chunk_size = 500
chunk_num = 0

print(f"Splitting into chunks of {chunk_size} features...")

# Iterate through batches
for batch in parquet_file.iter_batches(batch_size=chunk_size):
    chunk_num += 1

    # Convert to pandas
    df = batch.to_pandas()

    output_file = os.path.join(output_dir, f"lakes_chunk_{chunk_num:04d}.parquet")

    # Use the original schema when saving
    df.to_parquet(
        output_file,
        index=False,
        schema=schema  # ← This forces consistent schema!
    )
    print(f"Saved chunk {chunk_num}: {len(df)} features")

print(f"\n✅ Done! Created {chunk_num} files")