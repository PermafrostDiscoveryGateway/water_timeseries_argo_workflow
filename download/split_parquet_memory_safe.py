import pandas as pd
import pyarrow.parquet as pq
import os
from pathlib import Path

# Configuration
input_file = "/mnt/argo-filestore/water_timeseries/input/Nitze_etal_Lakes_filtered_full_set_V2d.parquet"
output_dir = "/mnt/argo-filestore/water_timeseries/input/split_lakes_1"
rows_per_file = 50000

# local config
# input_file = "/mnt/argo-filestore/water_timeseries/input/Nitze_etal_Lakes_filtered_full_set_V2d.parquet"
# output_dir = "/mnt/argo-filestore/water_timeseries/input/split_lakes_1"
# rows_per_file = 50000

# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)


def split_parquet_pyarrow(input_file, output_dir, rows_per_file):
    """Split parquet file using PyArrow (memory efficient)"""
    print(f"Reading parquet file: {input_file}")
    parquet_file = pq.ParquetFile(input_file)

    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows in file: {total_rows:,}")

    num_files = (total_rows + rows_per_file - 1) // rows_per_file
    print(f"Splitting into {num_files} files with ~{rows_per_file} rows each")

    # Process in batches
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=rows_per_file)):
        # Convert batch to table correctly
        table = pq.Table.from_batches([batch])

        # Write to new parquet file
        output_file = os.path.join(output_dir, f"part_{i:05d}.parquet")
        pq.write_table(table, output_file)

        print(f"Written {output_file} with {table.num_rows} rows")

    print(f"Done! Split into {num_files} files in {output_dir}")


def split_parquet_pandas_manual(input_file, output_dir, rows_per_file):
    """Split parquet file using pandas with manual chunking"""
    print(f"Reading parquet file in chunks: {input_file}")

    # First, read the entire file to get row count
    print("Getting total row count...")
    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows: {total_rows:,}")

    num_files = (total_rows + rows_per_file - 1) // rows_per_file
    print(f"Splitting into {num_files} files")

    # Process in chunks using row groups or range reading
    for i in range(num_files):
        start_row = i * rows_per_file
        end_row = min(start_row + rows_per_file, total_rows)

        # Read specific row range
        table = parquet_file.read_row_group(0)  # This doesn't work for ranges

        # Alternative: read the whole file and slice (not memory efficient)
        # Let's use a better approach - read in streaming fashion
        pass


def split_parquet_via_row_groups(input_file, output_dir, rows_per_file):
    """Split parquet file using row groups (most efficient)"""
    print(f"Reading parquet file: {input_file}")

    # Open the parquet file
    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows in file: {total_rows:,}")

    num_files = (total_rows + rows_per_file - 1) // rows_per_file
    print(f"Splitting into {num_files} files with ~{rows_per_file} rows each")

    # Collect batches
    batch_buffer = []
    current_batch_rows = 0
    file_counter = 0

    # Iterate through row groups
    for row_group_idx in range(parquet_file.num_row_groups):
        # Read one row group at a time
        table = parquet_file.read_row_group(row_group_idx)

        # Split the table into chunks of rows_per_file
        for start in range(0, table.num_rows, rows_per_file):
            end = min(start + rows_per_file, table.num_rows)
            chunk = table.slice(start, end - start)

            output_file = os.path.join(output_dir, f"part_{file_counter:05d}.parquet")
            pq.write_table(chunk, output_file)
            print(f"Written {output_file} with {chunk.num_rows} rows")
            file_counter += 1

    print(f"Done! Split into {file_counter} files in {output_dir}")


def split_parquet_simple_batch(input_file, output_dir, rows_per_file):
    """Simplest working approach - batch iterator"""
    print(f"Reading parquet file: {input_file}")

    # Open the file
    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows in file: {total_rows:,}")

    # Use iter_batches correctly
    batch_num = 0
    for batch in parquet_file.iter_batches(batch_size=rows_per_file):
        # Convert batch to table - this is the key fix
        table = pyarrow.Table.from_batches([batch])

        output_file = os.path.join(output_dir, f"part_{batch_num:05d}.parquet")
        pq.write_table(table, output_file)

        print(f"Written {output_file} with {len(batch)} rows")
        batch_num += 1

    print(f"Done! Split into {batch_num} files in {output_dir}")


# Most reliable method: use pandas with manual chunking via row iteration
def split_parquet_reliable(input_file, output_dir, rows_per_file):
    """Most reliable method - uses pandas but reads in chunks via row groups"""
    print(f"Opening parquet file: {input_file}")

    # Use pyarrow to read in chunks
    import pyarrow.parquet as pq
    import pyarrow as pa

    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows: {total_rows:,}")

    file_counter = 0
    rows_processed = 0

    # Read in batches
    for batch in parquet_file.iter_batches(batch_size=rows_per_file):
        # Convert RecordBatch to Table
        table = pa.Table.from_batches([batch])

        # Convert to pandas if you prefer (optional)
        # df = table.to_pandas()

        # Write to parquet
        output_file = os.path.join(output_dir, f"part_{file_counter:05d}.parquet")
        pq.write_table(table, output_file)

        rows_in_batch = len(batch)
        rows_processed += rows_in_batch
        print(f"Written {output_file} with {rows_in_batch} rows (total: {rows_processed:,}/{total_rows:,})")
        file_counter += 1

    print(f"✓ Done! Split {rows_processed:,} rows into {file_counter} files")


if __name__ == "__main__":
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        exit(1)

    # Import pyarrow here to ensure it's available
    import pyarrow as pa

    # Use the reliable method
    try:
        split_parquet_reliable(input_file, output_dir, rows_per_file)
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying alternative method...")

        # Fallback to simpler method
        import pyarrow.parquet as pq
        import pyarrow as pa

        try:
            # Read entire file (only if memory permits)
            print("Reading entire file (may use lots of memory)...")
            table = pq.read_table(input_file)
            total_rows = table.num_rows

            for i in range(0, total_rows, rows_per_file):
                end = min(i + rows_per_file, total_rows)
                chunk = table.slice(i, end - i)
                output_file = os.path.join(output_dir, f"part_{i // rows_per_file:05d}.parquet")
                pq.write_table(chunk, output_file)
                print(f"Written {output_file} with {chunk.num_rows} rows")

            print("Done!")
        except Exception as e2:
            print(f"Also failed: {e2}")
            print("\nPlease ensure you have the latest pyarrow installed:")
            print("pip install --upgrade pyarrow")