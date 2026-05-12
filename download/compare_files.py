import pyarrow.parquet as pq
import pandas as pd

# File paths
original_file = "/mnt/argo-filestore/water_timeseries/input/Nitze_etal_Lakes_filtered_full_set_V2d.parquet"
split_file = "/mnt/argo-filestore/water_timeseries/input/split_lakes_1/part_00080.parquet"


def compare_parquet_files(original, split):
    """Compare two parquet files for schema consistency"""

    print("=" * 80)
    print("COMPARING PARQUET FILES")
    print("=" * 80)
    print(f"Original: {original}")
    print(f"Split:    {split}")
    print()

    # Read metadata for both files
    print("Reading file metadata...")
    orig_metadata = pq.ParquetFile(original).metadata
    split_metadata = pq.ParquetFile(split).metadata

    # Read schemas
    orig_schema = pq.read_schema(original)
    split_schema = pq.read_schema(split)

    # 1. Compare column names
    print("\n" + "=" * 80)
    print("1. COLUMN NAMES COMPARISON")
    print("=" * 80)

    orig_columns = list(orig_schema.names)
    split_columns = list(split_schema.names)

    print(f"Original columns ({len(orig_columns)}): {orig_columns}")
    print(f"Split columns ({len(split_columns)}): {split_columns}")

    if orig_columns == split_columns:
        print("✓ Column names match exactly")
    else:
        print("✗ Column names DO NOT match!")

        # Find differences
        only_in_orig = set(orig_columns) - set(split_columns)
        only_in_split = set(split_columns) - set(orig_columns)

        if only_in_orig:
            print(f"  Columns only in original: {only_in_orig}")
        if only_in_split:
            print(f"  Columns only in split: {only_in_split}")

        # Check if order differs but sets are same
        if set(orig_columns) == set(split_columns):
            print("  (Column sets match but order differs)")

    # 2. Compare data types
    print("\n" + "=" * 80)
    print("2. DATA TYPES COMPARISON")
    print("=" * 80)

    all_types_match = True

    for col in orig_columns:
        if col in split_columns:
            orig_type = orig_schema.field(col).type
            split_type = split_schema.field(col).type

            match = orig_type == split_type
            all_types_match = all_types_match and match

            status = "✓" if match else "✗"
            print(f"{status} Column '{col}': Original={orig_type}, Split={split_type}")
        else:
            print(f"! Column '{col}' missing from split file")
            all_types_match = False

    if all_types_match:
        print("\n✓ All column data types match!")
    else:
        print("\n✗ Data type mismatches found!")

    # 3. Compare row counts
    print("\n" + "=" * 80)
    print("3. ROW COUNT COMPARISON")
    print("=" * 80)

    orig_total_rows = pq.read_table(original).num_rows

    # For split file, we need to know which rows it should contain
    # Assuming part_00080.parquet contains rows from index 80,000 to 80,999 (if 1000 per file)
    # Let's try to infer the row range from the filename
    import re
    part_num = int(re.search(r'part_(\d+)', split).group(1))
    rows_per_file = 50000  # Change this to match what you used

    start_row = part_num * rows_per_file
    end_row = min(start_row + rows_per_file, orig_total_rows)
    expected_rows = end_row - start_row

    split_rows = pq.read_table(split).num_rows

    print(f"Original total rows: {orig_total_rows:,}")
    print(f"Expected rows in {split}: {expected_rows:,} (rows {start_row:,} to {end_row - 1:,})")
    print(f"Actual rows in split: {split_rows:,}")

    if split_rows == expected_rows:
        print("✓ Row count matches expected number")
    else:
        print(f"✗ Row count mismatch! Expected {expected_rows:,}, got {split_rows:,}")

    # 4. Compare sample data (first and last few rows)
    print("\n" + "=" * 80)
    print("4. SAMPLE DATA COMPARISON")
    print("=" * 80)

    # Read original chunk that should correspond to split file
    orig_table = pq.read_table(original)
    orig_chunk = orig_table.slice(start_row, min(5, expected_rows))

    # Read split file
    split_table = pq.read_table(split)
    split_sample = split_table.slice(0, min(5, split_rows))

    print(f"\nFirst 5 rows from ORIGINAL (rows {start_row}-{start_row + 4}):")
    print(orig_chunk.to_pandas().head())

    print(f"\nFirst 5 rows from SPLIT file:")
    print(split_sample.to_pandas().head())

    # 5. Check for unique ID column
    print("\n" + "=" * 80)
    print("5. UNIQUE ID COLUMN CHECK")
    print("=" * 80)

    # Look for common ID column names
    possible_id_columns = ['id', 'ID', 'lake_id', 'Lake_ID', 'unique_id', 'Unique_ID', 'index']
    id_columns_found = [col for col in orig_columns if any(id_name in col.lower() for id_name in ['id', 'index'])]

    if id_columns_found:
        print(f"Found potential ID columns: {id_columns_found}")

        # Check if IDs are unique in split file
        for id_col in id_columns_found:
            if id_col in split_columns:
                split_df = split_table.to_pandas()
                unique_count = split_df[id_col].nunique()
                total_count = len(split_df)

                if unique_count == total_count:
                    print(f"✓ Column '{id_col}' has all unique values in split file ({unique_count}/{total_count})")
                else:
                    print(f"! Column '{id_col}' has {unique_count} unique values out of {total_count} total rows")
    else:
        print("No obvious ID column found. Check column names manually.")

    # 6. Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if orig_columns == split_columns and all_types_match and split_rows == expected_rows:
        print("✓✓✓ ALL CHECKS PASSED! Split file is consistent with original.")
    else:
        print("✗✗✗ SOME CHECKS FAILED! Review issues above.")

    print("\nDetailed results:")
    print(f"  Column names match: {orig_columns == split_columns}")
    print(f"  Data types match: {all_types_match}")
    print(f"  Row count matches expected: {split_rows == expected_rows}")


def quick_compare_with_pandas(original, split):
    """Simpler comparison using pandas"""
    print("\n" + "=" * 80)
    print("QUICK COMPARISON WITH PANDAS")
    print("=" * 80)

    # Read both files (only necessary rows from original)
    # Extract part number to know which rows to read
    import re
    part_num = int(re.search(r'part_(\d+)', split).group(1))
    rows_per_file = 1000  # Adjust if you changed this

    # Read original file but only the relevant chunk
    orig_df = pd.read_parquet(original)
    start_row = part_num * rows_per_file
    end_row = min(start_row + rows_per_file, len(orig_df))
    orig_chunk = orig_df.iloc[start_row:end_row]

    # Read split file
    split_df = pd.read_parquet(split)

    print(f"\nOriginal chunk shape: {orig_chunk.shape}")
    print(f"Split file shape: {split_df.shape}")

    # Compare dtypes
    print("\nData types comparison:")
    dtype_comparison = pd.DataFrame({
        'Original_dtype': orig_chunk.dtypes,
        'Split_dtype': split_df.dtypes
    })
    print(dtype_comparison)

    # Check if dataframes are identical
    if orig_chunk.shape == split_df.shape:
        # Try to compare values (may be slow for large files)
        if orig_chunk.equals(split_df):
            print("\n✓ Data is identical!")
        else:
            print("\n✗ Data differs (check column order or values)")
    else:
        print("\n✗ Shapes don't match")


if __name__ == "__main__":
    # Run main comparison
    compare_parquet_files(original_file, split_file)

    # Optional: Run pandas-based quick comparison
    print("\n" + "=" * 80)
    response = input("Run pandas quick comparison? (y/n): ")
    if response.lower() == 'y':
        quick_compare_with_pandas(original_file, split_file)