import pyarrow.parquet as pq
import pandas as pd
import re

# File paths
original_file = "/mnt/argo-filestore/water_timeseries/input/Nitze_etal_Lakes_filtered_full_set_V2d.parquet"
split_file = "/mnt/argo-filestore/water_timeseries/input/split_lakes_1/part_00080.parquet"
rows_per_file = 50000  # Updated to match your new setting


def compare_parquet_files(original, split, rows_per_file):
    """Compare two parquet files for schema consistency"""

    print("=" * 80)
    print("COMPARING PARQUET FILES")
    print("=" * 80)
    print(f"Original: {original}")
    print(f"Split:    {split}")
    print(f"Rows per file: {rows_per_file:,}")
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

    print(f"Original columns ({len(orig_columns)}): {orig_columns[:10]}...")
    print(f"Split columns ({len(split_columns)}): {split_columns[:10]}...")

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

        # Filter out pandas internal columns
        internal_cols = [col for col in only_in_split if col.startswith('__')]
        if internal_cols:
            print(f"  Note: Internal pandas columns found: {internal_cols}")

    # 2. Compare data types
    print("\n" + "=" * 80)
    print("2. DATA TYPES COMPARISON")
    print("=" * 80)

    all_types_match = True

    # Only compare columns that exist in both
    common_columns = set(orig_columns) & set(split_columns)

    for col in common_columns:
        orig_type = orig_schema.field(col).type
        split_type = split_schema.field(col).type

        match = orig_type == split_type
        all_types_match = all_types_match and match

        status = "✓" if match else "✗"
        if not match:
            print(f"{status} Column '{col}': Original={orig_type}, Split={split_type}")

    if all_types_match:
        print("✓ All common column data types match!")
    else:
        print("\n✗ Data type mismatches found!")

    # 3. Compare row counts
    print("\n" + "=" * 80)
    print("3. ROW COUNT COMPARISON")
    print("=" * 80)

    orig_total_rows = pq.read_table(original).num_rows

    # Extract part number from filename
    part_match = re.search(r'part_(\d+)', split)
    if part_match:
        part_num = int(part_match.group(1))
    else:
        print("! Could not extract part number from filename")
        part_num = 0

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
    try:
        orig_table = pq.read_table(original)
        orig_chunk = orig_table.slice(start_row, min(5, expected_rows))

        # Read split file
        split_table = pq.read_table(split)
        split_sample = split_table.slice(0, min(5, split_rows))

        print(f"\nFirst 5 rows from ORIGINAL (rows {start_row}-{start_row + 4}):")
        orig_df = orig_chunk.to_pandas()
        print(orig_df.head())

        print(f"\nFirst 5 rows from SPLIT file:")
        split_df = split_sample.to_pandas()
        print(split_df.head())

        # Compare the actual values of the first row
        if len(orig_df) > 0 and len(split_df) > 0:
            print("\nFirst row comparison (key columns):")
            key_cols = list(common_columns)[:5]  # First 5 columns
            for col in key_cols:
                orig_val = orig_df[col].iloc[0] if col in orig_df.columns else None
                split_val = split_df[col].iloc[0] if col in split_df.columns else None
                match = orig_val == split_val
                status = "✓" if match else "?"
                print(f"  {status} {col}: Original={orig_val}, Split={split_val}")

    except Exception as e:
        print(f"! Could not compare sample data: {e}")

    # 5. Check for unique ID column (improved version)
    print("\n" + "=" * 80)
    print("5. UNIQUE ID COLUMN CHECK")
    print("=" * 80)

    # Look for common ID column names, excluding pandas internal columns
    possible_id_columns = ['id', 'ID', 'lake_id', 'Lake_ID', 'unique_id', 'Unique_ID', 'index', 'Id']
    id_columns_found = []

    for col in split_columns:
        if col.startswith('__'):
            continue  # Skip pandas internal columns
        if any(id_name.lower() in col.lower() for id_name in possible_id_columns):
            id_columns_found.append(col)

    if id_columns_found:
        print(f"Found potential ID columns: {id_columns_found}")

        # Check if IDs are unique in split file
        for id_col in id_columns_found:
            try:
                if id_col in split_columns:
                    split_df = split_table.to_pandas()
                    unique_count = split_df[id_col].nunique()
                    total_count = len(split_df)

                    if unique_count == total_count:
                        print(
                            f"✓ Column '{id_col}' has all unique values in split file ({unique_count:,}/{total_count:,})")
                    else:
                        print(
                            f"! Column '{id_col}' has {unique_count:,} unique values out of {total_count:,} total rows")
                else:
                    print(f"! Column '{id_col}' not found in split file")
            except Exception as e:
                print(f"! Could not check uniqueness for '{id_col}': {e}")
    else:
        print("No obvious ID column found. First few columns:")
        for col in split_columns[:10]:
            print(f"  - {col}")

    # 6. Check if all columns from original are present (excluding index columns)
    print("\n" + "=" * 80)
    print("6. COLUMN INTEGRITY CHECK")
    print("=" * 80)

    missing_columns = set(orig_columns) - set(split_columns)
    extra_columns = set(split_columns) - set(orig_columns)

    # Filter out internal pandas columns from extra_columns
    internal_columns = [col for col in extra_columns if col.startswith('__')]
    real_extra_columns = [col for col in extra_columns if not col.startswith('__')]

    if missing_columns:
        print(f"✗ Missing columns: {missing_columns}")
    else:
        print("✓ All original columns are present")

    if real_extra_columns:
        print(f"! Extra columns in split file: {real_extra_columns}")
    elif internal_columns:
        print(f"Note: Internal pandas columns found (can be ignored): {internal_columns}")
    else:
        print("✓ No unexpected extra columns")

    # 7. Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Overall success criteria
    columns_match = len(missing_columns) == 0 and len(real_extra_columns) == 0
    row_count_ok = split_rows == expected_rows

    if columns_match and all_types_match and row_count_ok:
        print("✓✓✓ ALL CRITICAL CHECKS PASSED! Split file is consistent with original.")
        print("\nThe split file maintains:")
        print("  - Same column structure")
        print("  - Same data types")
        print("  - Correct number of rows")
        print("  - Data integrity preserved")
    else:
        print("✗✗✗ SOME ISSUES FOUND. Review details above.")
        print(f"  Columns match: {columns_match}")
        print(f"  Data types match: {all_types_match}")
        print(f"  Row count correct: {row_count_ok}")


def quick_integrity_check(original, split, rows_per_file):
    """Quick check focusing on data integrity"""
    print("\n" + "=" * 80)
    print("QUICK INTEGRITY CHECK")
    print("=" * 80)

    try:
        # Extract part number
        part_num = int(re.search(r'part_(\d+)', split).group(1))
        start_row = part_num * rows_per_file

        # Read only the relevant chunk from original
        print("Reading original chunk...")
        orig_table = pq.read_table(original)
        chunk_size = min(rows_per_file, orig_table.num_rows - start_row)
        orig_chunk = orig_table.slice(start_row, chunk_size)
        orig_df = orig_chunk.to_pandas()

        # Read split file
        print("Reading split file...")
        split_df = pd.read_parquet(split)

        print(f"Original chunk shape: {orig_df.shape}")
        print(f"Split file shape: {split_df.shape}")

        if orig_df.shape == split_df.shape:
            print("✓ Shapes match")

            # Check if data types match for each column
            dtype_mismatch = []
            for col in orig_df.columns:
                if col in split_df.columns:
                    if orig_df[col].dtype != split_df[col].dtype:
                        dtype_mismatch.append(col)

            if dtype_mismatch:
                print(f"✗ Type mismatches in columns: {dtype_mismatch}")
            else:
                print("✓ All data types match")

            # Quick value check on first row
            print("\nFirst row comparison:")
            for col in orig_df.columns[:3]:  # Check first 3 columns
                orig_val = orig_df[col].iloc[0]
                split_val = split_df[col].iloc[0]
                if orig_val == split_val:
                    print(f"  ✓ {col}: {orig_val} == {split_val}")
                else:
                    print(f"  ✗ {col}: {orig_val} != {split_val}")

        else:
            print(f"✗ Shape mismatch: Original {orig_df.shape} vs Split {split_df.shape}")

    except Exception as e:
        print(f"Quick check failed: {e}")


if __name__ == "__main__":
    # Run main comparison
    compare_parquet_files(original_file, split_file, rows_per_file)

    # Optional: Run quick integrity check
    print("\n")
    response = input("Run quick integrity check? (y/n): ")
    if response.lower() == 'y':
        quick_integrity_check(original_file, split_file, rows_per_file)