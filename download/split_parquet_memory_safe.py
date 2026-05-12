import pandas as pd
import pyarrow.parquet as pq
import os
from pathlib import Path

# Configuration
input_file = "/mnt/argo-filestore/water_timeseries/input/Nitze_etal_Lakes_filtered_full_set_V2d.parquet"
output_dir = "/mnt/argo-filestore/water_timeseries/input/split_lakes_1"
rows_per_file = 1000

# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)


# Method 1: Using PyArrow (more memory efficient for very large files)
def split_parquet_pyarrow(input_file, output_dir, rows_per_file):
    print(f"Reading parquet file: {input_file}")
    parquet_file = pq.ParquetFile(input_file)

    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows in file: {total_rows:,}")

    num_files = (total_rows + rows_per_file - 1) // rows_per_file
    print(f"Splitting into {num_files} files with ~{rows_per_file} rows each")

    # Process in batches
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=rows_per_file)):
        # Convert batch to table
        table = batch.to_table()

        # Write to new parquet file
        output_file = os.path.join(output_dir, f"part_{i:05d}.parquet")
        pq.write_table(table, output_file)

        print(f"Written {output_file} with {table.num_rows} rows")

    print(f"Done! Split into {num_files} files in {output_dir}")


# Method 2: Using pandas (simpler but may use more memory)
def split_parquet_pandas(input_file, output_dir, rows_per_file):
    print(f"Reading parquet file: {input_file}")

    # Read in chunks to avoid memory issues
    chunk_iter = pd.read_parquet(input_file, chunksize=rows_per_file)

    file_count = 0
    total_rows = 0

    for chunk in chunk_iter:
        output_file = os.path.join(output_dir, f"part_{file_count:05d}.parquet")
        chunk.to_parquet(output_file, index=False)

        rows_in_chunk = len(chunk)
        total_rows += rows_in_chunk
        print(f"Written {output_file} with {rows_in_chunk} rows")
        file_count += 1

    print(f"Done! Split {total_rows:,} rows into {file_count} files in {output_dir}")


# Choose which method to use:
# - PyArrow is generally more memory efficient
# - Pandas is simpler but may use more RAM

if __name__ == "__main__":
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        exit(1)

    # Use PyArrow method (recommended for very large files)
    try:
        split_parquet_pyarrow(input_file, output_dir, rows_per_file)
    except Exception as e:
        print(f"PyArrow method failed: {e}")
        print("Trying pandas method...")
        split_parquet_pandas(input_file, output_dir, rows_per_file)