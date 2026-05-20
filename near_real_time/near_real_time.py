from loguru import logger
import os
import glob
import sys
from dotenv import load_dotenv
import datetime
import download_new_dynamic_world_data
from water_timeseries.breakpoint import NRTBreakpoint
from water_timeseries.dataset import DWDataset
import xarray as xr
import pandas as pd
import dask.dataframe as dd
from pathlib import Path


# Add this comprehensive comparison function before the return statement:

def robust_dataframe_comparison(df1, df2, name1="Old", name2="Parquet"):
    """Robust comparison handling index and column ordering"""
    print(f"\n{'=' * 70}")
    print(f"ROBUST DATAFRAME COMPARISON: {name1} vs {name2}")
    print(f"{'=' * 70}")

    # 1. Compare shapes
    print(f"\n1. Shape comparison:")
    print(f"   {name1}: {df1.shape}")
    print(f"   {name2}: {df2.shape}")
    if df1.shape != df2.shape:
        print(f"   ❌ Shapes differ!")
        return False

    # 2. Compare columns (ignoring order)
    print(f"\n2. Column comparison:")
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    if cols1 == cols2:
        print(f"   ✅ Column sets match")
        if list(df1.columns) != list(df2.columns):
            print(f"   ℹ️  Column order differs but content is the same")
            # Reorder df2 to match df1's column order for comparison
            df2 = df2[df1.columns]
    else:
        print(f"   ❌ Column sets differ")
        print(f"   Only in {name1}: {cols1 - cols2}")
        print(f"   Only in {name2}: {cols2 - cols1}")
        return False

    # 3. Handle index comparison properly
    print(f"\n3. Index comparison:")
    print(f"   {name1} index type: {type(df1.index)}")
    print(f"   {name2} index type: {type(df2.index)}")

    # Convert indexes to lists for comparison
    idx1_list = list(df1.index)
    idx2_list = list(df2.index)

    if idx1_list == idx2_list:
        print(f"   ✅ Indexes match exactly (same order)")
        sort_needed = False
    else:
        print(f"   ⚠️ Indexes differ in order or content")
        # Check if they have same values but different order
        if set(idx1_list) == set(idx2_list):
            print(f"   ℹ️  Index values are the same but in different order")
            print(f"   Will sort both DataFrames by index for comparison")
            sort_needed = True
        else:
            print(f"   ❌ Index values differ")
            print(f"   Only in {name1}: {set(idx1_list) - set(idx2_list)}")
            print(f"   Only in {name2}: {set(idx2_list) - set(idx1_list)}")
            return False

    # 4. Sort if needed and compare values
    print(f"\n4. Value comparison:")
    df1_comp = df1.sort_index() if sort_needed else df1
    df2_comp = df2.sort_index() if sort_needed else df2

    # Check if indexes now match after sorting
    if not df1_comp.index.equals(df2_comp.index):
        print(f"   ❌ Indexes still don't match after sorting")
        return False

    # Compare each column with tolerance for floats
    all_match = True
    tolerance = 1e-6

    for col in df1_comp.columns:
        col1 = df1_comp[col]
        col2 = df2_comp[col]

        # Check dtype
        if col1.dtype != col2.dtype:
            print(f"   ⚠️ Column '{col}' dtype mismatch: {col1.dtype} vs {col2.dtype}")

        # Compare values
        if col1.dtype == 'datetime64[ns]':
            match = col1.equals(col2)
        elif col1.dtype in ['float64', 'float32']:
            # Use tolerance for floats
            diff = abs(col1 - col2)
            max_diff = diff.max()
            match = max_diff < tolerance if not pd.isna(max_diff) else col1.isna().all() and col2.isna().all()
            if not match and max_diff >= tolerance:
                print(f"   ❌ Column '{col}' differs (max diff: {max_diff:.2e})")
                # Show sample differences
                diff_mask = diff > tolerance
                if diff_mask.any():
                    sample_indices = diff_mask[diff_mask].index[:3]
                    for idx in sample_indices:
                        print(f"      {idx}: {name1}={col1.loc[idx]:.6f}, {name2}={col2.loc[idx]:.6f}")
        else:
            match = col1.equals(col2)

        if not match:
            all_match = False
            if 'max_diff' not in locals() or max_diff >= tolerance:
                print(f"   ❌ Column '{col}' values differ")

    # 5. Memory analysis
    print(f"\n5. Memory analysis:")
    mem1 = df1.memory_usage(deep=True)
    mem2 = df2.memory_usage(deep=True)

    print(f"   Total memory:")
    print(f"     {name1}: {mem1.sum() / 1024 ** 2:.2f} MB")
    print(f"     {name2}: {mem2.sum() / 1024 ** 2:.2f} MB")

    # Check index memory specifically
    idx_mem1 = df1.index.memory_usage(deep=True)
    idx_mem2 = df2.index.memory_usage(deep=True)
    print(f"\n   Index memory:")
    print(f"     {name1}: {idx_mem1 / 1024 ** 2:.3f} MB")
    print(f"     {name2}: {idx_mem2 / 1024 ** 2:.3f} MB")

    # 6. Final verdict
    print(f"\n{'=' * 70}")
    if all_match:
        print(f"✅ VERDICT: DataFrames are IDENTICAL in content")
        print(f"   Memory difference of {abs(mem1.sum() - mem2.sum()) / 1024 ** 2:.2f} MB is due to:")
        if idx_mem1 != idx_mem2:
            print(f"     - Index memory optimization (saved {abs(idx_mem1 - idx_mem2) / 1024 ** 2:.3f} MB)")
        print(f"     - Internal pandas vs pyarrow storage optimizations")
        print(f"   ✅ This is NORMAL and EXPECTED - parquet method is more memory efficient")
        return True
    else:
        print(f"❌ VERDICT: DataFrames have DIFFERENT content")
        print(f"   Investigate the differences shown above")
        return False


    # Replace the old comparison code with this call:
    print("\n=== RUNNING ROBUST DATAFRAME COMPARISON ===")
    are_identical = robust_dataframe_comparison(
        final_results_old_method,
        final_results_from_parquet,
        "Old Method (in-memory concat)",
        "Parquet Method (Dask)"
    )

    if are_identical:
        print("\n✅ Both methods produce the same results. The Dask/parquet approach is working correctly!")
        print("   The memory difference is just due to more efficient storage in PyArrow.")
    else:
        print("\n⚠️ Differences detected. Check the output above for details.")

def precompute_nrt_breakpoints(
        input_nc_file: str | Path,
        output_dir: str | Path,
        lake_chunk_size: int = 500,
        analysis_date: str | pd.Timestamp | None = None,
        data_aggregation_period: str = "monthly"
) -> pd.DataFrame:
    input_nc_file = Path(input_nc_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # FIRST: Get lake IDs without loading full dataset
    print("Reading lake IDs from file...")
    with xr.open_dataset(input_nc_file, engine='netcdf4') as ds_meta:
        all_lake_ids = ds_meta.id_geohash.values
        total_lakes = len(all_lake_ids)
        print(f"Total lakes: {total_lakes}")

        # Convert analysis_date to datetime if needed
        if analysis_date is None:
            dates = ds_meta.date.values
            analysis_date = pd.to_datetime(dates[-3])
            print(f"Using most recent date: {analysis_date}")
        else:
            # Convert string to datetime if necessary
            if isinstance(analysis_date, str):
                analysis_date = pd.to_datetime(analysis_date)
                print(f"Using specified analysis date: {analysis_date}")

            # Verify the date exists in the dataset
            if analysis_date not in ds_meta.date.values:
                print(f"Warning: {analysis_date} not in dataset dates")
                dates = ds_meta.date.values
                print(f"Available date range: {dates[0]} to {dates[-1]}")
                # Use the most recent date instead
                analysis_date = pd.to_datetime(dates[-1])
                print(f"Using most recent date instead: {analysis_date}")
    analysis_date_string = str(analysis_date)
    analysis_date_string = analysis_date_string.split(" ")[0].replace('-', '_')
    chunk_output_subdir_name = f"chunk_output_{analysis_date_string}"
    chunk_output_dir = os.path.join(output_dir, chunk_output_subdir_name)
    chunk_output_dir = Path(chunk_output_dir)
    chunk_output_dir.mkdir(parents=True, exist_ok=True)

    results = []  # Keep this for comparison

    # Process each chunk
    for i in range(0, total_lakes, lake_chunk_size):
        chunk_ids = all_lake_ids[i:i + lake_chunk_size]
        print(f"\nProcessing chunk {i // lake_chunk_size + 1}/{(total_lakes + lake_chunk_size - 1) // lake_chunk_size}")

        # Load chunk
        ds = xr.open_dataset(
            input_nc_file,
            engine='netcdf4',
            chunks={'id_geohash': lake_chunk_size, 'date': -1}
        )
        ds_chunk = ds.sel(id_geohash=chunk_ids).load()
        ds.close()

        # Verify the chunk has data for analysis_date
        if analysis_date not in ds_chunk.date.values:
            print(f"  Warning: analysis_date {analysis_date} not in this chunk's dates")
            print(f"  Chunk date range: {ds_chunk.date.values[0]} to {ds_chunk.date.values[-1]}")
            # Use the most recent date from this chunk
            chunk_analysis_date = pd.to_datetime(ds_chunk.date.values[-1])
            print(f"  Using chunk's most recent date: {chunk_analysis_date}")
        else:
            chunk_analysis_date = analysis_date

        # Check if chunk has valid lakes for this date
        ds_analysis_test = ds_chunk.sel(date=chunk_analysis_date)
        valid_lakes = ds_analysis_test.dropna(dim="id_geohash", how="all").id_geohash.values

        if len(valid_lakes) == 0:
            print(f"  No valid lakes for date {chunk_analysis_date}, skipping chunk")
            ds_chunk.close()
            del ds_chunk
            continue

        print(f"  Found {len(valid_lakes)} valid lakes for date {chunk_analysis_date}")

        # Create dataset wrapper for just this chunk
        dw_dataset_chunk = DWDataset(ds_chunk)

        # Initialize NRT breakpoint detector
        nrt_breakpoint = NRTBreakpoint(kwargs_break={})

        # Calculate breakpoints for this chunk
        chunk_output_file = chunk_output_dir / f"nrt_results_chunk_{i // lake_chunk_size + 1}.parquet"
        chunk_output_file_exists = os.path.exists(chunk_output_file)
        logger.debug(f"  Chunk output file {chunk_output_file} exists: {chunk_output_file_exists}")
        if not chunk_output_file_exists:
            try:
                chunk_result = nrt_breakpoint.calculate_break(
                    dataset=dw_dataset_chunk,
                    analysis_date=chunk_analysis_date,  # Use the datetime64 version
                    data_aggregation_period=data_aggregation_period,
                    object_id=chunk_ids,
                )

                if chunk_result is not None and len(chunk_result) > 0:
                    # results.append(chunk_result)  # Keep for comparison

                    # Save chunk results
                    chunk_output_file = chunk_output_dir / f"nrt_results_chunk_{i // lake_chunk_size + 1}.parquet"
                    chunk_result.to_parquet(chunk_output_file, index=True)
                    print(f"  Saved chunk results to {chunk_output_file}")
                else:
                    print(f"  No results generated for chunk")

            except ValueError as e:
                if "n_jobs == 0" in str(e):
                    print(f"  No valid lakes for analysis in this chunk")
                else:
                    print(f"  Error processing chunk: {e}")
                continue

            # Clear memory
            ds_chunk.close()
            del ds_chunk
            del dw_dataset_chunk
            import gc
            gc.collect()

    # Combine results from parquet files using Dask (memory-efficient)
    final_output_file_name_from_parquet = "nrt_breakpoints_all_lakes_from_parquet_" + analysis_date_string + ".parquet"
    final_output_file_from_parquet = output_dir / final_output_file_name_from_parquet

    parquet_files = sorted(chunk_output_dir.glob("nrt_results_chunk_*.parquet"))

    if parquet_files:
        print(f"\n📂 Found {len(parquet_files)} chunk parquet files to combine")

        # Use Dask to read all parquet files lazily
        # This doesn't load data into memory immediately
        print("  Using Dask for out-of-core processing...")
        ddf = dd.read_parquet(
            str(chunk_output_dir / "nrt_results_chunk_*.parquet"),
            engine='pyarrow'  # Use pyarrow engine for better performance
        )

        # Show the Dask task graph info
        print(f"  Dask DataFrame partitions: {ddf.npartitions}")
        print(f"  Dask DataFrame columns: {list(ddf.columns)}")

        # Compute the result (this is where data is actually loaded and combined)
        # Dask handles the chunking and memory management automatically
        print("  Computing combined DataFrame (this may take a moment)...")
        final_results_from_parquet = ddf.compute()

        # Save combined results
        final_results_from_parquet.to_parquet(final_output_file_from_parquet, index=True)
        print(f"\n✅ Combined results from parquet files saved to {final_output_file_from_parquet}")
        print(f"   Total rows: {len(final_results_from_parquet):,}")
        print(f"   Memory usage: {final_results_from_parquet.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        # Optional: Save as CSV as well for easy inspection (commented out)
        # csv_output_file = output_dir / f"nrt_breakpoints_all_lakes_{analysis_date_string}.csv"
        # final_results_from_parquet.to_csv(csv_output_file, index=False)
        # print(f"   Also saved as CSV: {csv_output_file}")

    else:
        print("⚠ No parquet files found to combine")
        final_results_from_parquet = pd.DataFrame()

    # OLD METHOD: Combine from results list (keep for comparison)
    if results:
        print(f"\n📊 Old method collected {len(results)} result chunks in memory")
        final_results_old_method = pd.concat(results, axis=0)
        final_output_file_old = output_dir / f"nrt_breakpoints_all_lakes_old_method_{analysis_date_string}.parquet"
        final_results_old_method.to_parquet(final_output_file_old, index=True)
        print(f"✅ Old method results saved to {final_output_file_old}")
        print(f"   Total rows: {len(final_results_old_method):,}")

        # Compare the two results if both methods produced data
        if not final_results_from_parquet.empty:
            if len(final_results_old_method) == len(final_results_from_parquet):
                print(f"\n✅ Results match: both methods produced {len(final_results_from_parquet):,} rows")
            else:
                print(
                    f"\n⚠️ Results size mismatch: Old method={len(final_results_old_method):,}, Parquet method={len(final_results_from_parquet):,}")

            # Comprehensive comparison of DataFrames
            print("\n=== DETAILED DATAFRAME COMPARISON ===")
            compare = robust_dataframe_comparison(final_results_old_method, final_results_from_parquet)

            # 1. Compare column names and order
            print("\n1. Column comparison:")
            old_cols = list(final_results_old_method.columns)
            parquet_cols = list(final_results_from_parquet.columns)

            if old_cols == parquet_cols:
                print(f"   ✅ Column names and order match: {old_cols}")
            else:
                print(f"   ⚠️ Column mismatch!")
                print(f"   Old method columns: {old_cols}")
                print(f"   Parquet method columns: {parquet_cols}")
                print(f"   Columns only in old: {set(old_cols) - set(parquet_cols)}")
                print(f"   Columns only in parquet: {set(parquet_cols) - set(old_cols)}")

            # 2. Compare data types
            print("\n2. Data type comparison:")
            dtype_comparison = []
            for col in set(old_cols) & set(parquet_cols):
                old_dtype = final_results_old_method[col].dtype
                parquet_dtype = final_results_from_parquet[col].dtype
                if old_dtype == parquet_dtype:
                    print(f"   ✅ {col}: {old_dtype} == {parquet_dtype}")
                else:
                    print(f"   ⚠️ {col}: {old_dtype} != {parquet_dtype}")
                    dtype_comparison.append((col, old_dtype, parquet_dtype))

            # 3. Check for index differences
            print("\n3. Index comparison:")
            if final_results_old_method.index.equals(final_results_from_parquet.index):
                print(f"   ✅ Indexes are identical")
            else:
                print(f"   ⚠️ Indexes differ")
                print(f"   Old method index type: {type(final_results_old_method.index)}")
                print(f"   Parquet method index type: {type(final_results_from_parquet.index)}")
                print(f"   Old method index sample: {final_results_old_method.index[:5].tolist()}")
                print(f"   Parquet method index sample: {final_results_from_parquet.index[:5].tolist()}")

            # 4. Check for null values
            print("\n4. Null value comparison:")
            old_nulls = final_results_old_method.isnull().sum()
            parquet_nulls = final_results_from_parquet.isnull().sum()

            if old_nulls.equals(parquet_nulls):
                print(f"   ✅ Null value counts match")
            else:
                print(f"   ⚠️ Null value count mismatch:")
                for col in set(old_cols) & set(parquet_cols):
                    old_null = old_nulls[col] if col in old_nulls else 0
                    parquet_null = parquet_nulls[col] if col in parquet_nulls else 0
                    if old_null != parquet_null:
                        print(f"      {col}: Old={old_null}, Parquet={parquet_null}")

            # 5. Sample row comparison (first 5 rows)
            print("\n5. Sample data comparison (first 5 rows):")
            print("\n   Old method - first 5 rows:")
            print(final_results_old_method.head())
            print("\n   Parquet method - first 5 rows:")
            print(final_results_from_parquet.head())

            # 6. Statistical summary comparison for numeric columns
            print("\n6. Statistical comparison for numeric columns:")
            numeric_cols = final_results_old_method.select_dtypes(include=['number']).columns
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                if col in final_results_from_parquet.columns:
                    old_stats = final_results_old_method[col].describe()
                    parquet_stats = final_results_from_parquet[col].describe()
                    print(f"\n   Column: {col}")
                    print(
                        f"   Old: mean={old_stats['mean']:.3f}, std={old_stats['std']:.3f}, min={old_stats['min']:.3f}, max={old_stats['max']:.3f}")
                    print(
                        f"   Parquet: mean={parquet_stats['mean']:.3f}, std={parquet_stats['std']:.3f}, min={parquet_stats['min']:.3f}, max={parquet_stats['max']:.3f}")

            # 7. Check for duplicate rows or index values
            print("\n7. Duplicate check:")
            old_duplicates = final_results_old_method.duplicated().sum()
            parquet_duplicates = final_results_from_parquet.duplicated().sum()
            print(f"   Old method duplicates: {old_duplicates}")
            print(f"   Parquet method duplicates: {parquet_duplicates}")

            # 8. Check row order differences (if rows are shuffled)
            print("\n8. Row order comparison (first 5 rows after sorting by index):")
            try:
                old_sorted = final_results_old_method.sort_index()
                parquet_sorted = final_results_from_parquet.sort_index()
                if old_sorted.head().equals(parquet_sorted.head()):
                    print("   ✅ Rows match after sorting by index")
                else:
                    print("   ⚠️ Rows differ even after sorting by index")
                    # Try to find if it's just a sorting issue
                    common_cols = set(old_cols) & set(parquet_cols)
                    if common_cols:
                        # Check if values are the same but order is different
                        old_values = old_sorted[list(common_cols)].values
                        parquet_values = parquet_sorted[list(common_cols)].values
                        if set(map(tuple, old_values)) == set(map(tuple, parquet_values)):
                            print("   ℹ️  Data content is the same, but rows are in different order")
                        else:
                            print("   ⚠️  Data content differs")
            except Exception as e:
                print(f"   Could not compare sorted rows: {e}")


            # Add this after the memory usage comparison section (replace that section with this):

            # 9. Memory usage comparison with detailed breakdown
            print("\n9. Detailed memory usage comparison:")
            old_memory = final_results_old_method.memory_usage(deep=True)
            parquet_memory = final_results_from_parquet.memory_usage(deep=True)

            print(f"   Total memory:")
            print(f"     Old method: {old_memory.sum() / 1024 ** 2:.2f} MB")
            print(f"     Parquet method: {parquet_memory.sum() / 1024 ** 2:.2f} MB")
            print(f"     Difference: {abs(old_memory.sum() - parquet_memory.sum()) / 1024 ** 2:.2f} MB")

            print(f"\n   Memory per column (MB):")
            for col in old_cols:
                old_col_mem = old_memory[col] / 1024 ** 2
                parquet_col_mem = parquet_memory[col] / 1024 ** 2 if col in parquet_memory else 0
                diff_mb = abs(old_col_mem - parquet_col_mem)
                if diff_mb > 0.01:  # Only show differences > 10KB
                    print(f"     {col}: Old={old_col_mem:.3f}, Parquet={parquet_col_mem:.3f}, Diff={diff_mb:.3f}")

            # Check for categorical vs object dtypes (major memory difference source)
            print(f"\n   Memory optimization check:")
            for col in old_cols:
                old_dtype = final_results_old_method[col].dtype
                parquet_dtype = final_results_from_parquet[col].dtype
                if old_dtype != parquet_dtype:
                    print(f"     {col}: dtype mismatch - {old_dtype} vs {parquet_dtype}")

            # Check string columns for object vs string[pyarrow] differences
            string_cols = [col for col in old_cols if final_results_old_method[col].dtype == 'object']
            if string_cols:
                print(f"\n   String columns (object dtype): {string_cols}")
                print(f"     These may be more memory efficient as 'string[pyarrow]' dtype")

            # Check if index memory usage differs
            old_index_memory = final_results_old_method.index.memory_usage(deep=True)
            parquet_index_memory = final_results_from_parquet.index.memory_usage(deep=True)
            print(f"\n   Index memory usage:")
            print(f"     Old method index: {old_index_memory / 1024 ** 2:.3f} MB")
            print(f"     Parquet method index: {parquet_index_memory / 1024 ** 2:.3f} MB")

            # Potential explanations
            print(f"\n   Potential explanations for memory difference:")
            print(f"     1. String storage optimization (pyarrow vs pandas)")
            print(f"     2. Different chunk sizes when reading parquet files")
            print(f"     3. Categorical column conversion")
            print(f"     4. Index memory optimization")
            print(f"     5. Different NaN representation")

            # Check if data is actually identical (ignoring memory)
            print(f"\n   Data equality check (ignoring memory layout):")
            try:
                # Sort by index to compare values
                old_sorted = final_results_old_method.sort_index()
                parquet_sorted = final_results_from_parquet.sort_index()

                # Compare values (not memory)
                if old_sorted.equals(parquet_sorted):
                    print(f"     ✅ Data content is identical!")
                    print(f"     ✅ Memory difference is just due to internal representation/optimization")
                    print(f"     ✅ This is normal and expected when reading from parquet files")
                else:
                    print(f"     ⚠️ Data content differs (beyond just memory layout)")

                    # Find where differences occur
                    diff_mask = (old_sorted != parquet_sorted)
                    diff_count = diff_mask.sum().sum()
                    if diff_count > 0:
                        print(f"     Number of differing values: {diff_count}")
                        # Show first few differences
                        for col in old_cols:
                            col_diff = diff_mask[col].sum()
                            if col_diff > 0:
                                print(f"       {col}: {col_diff} differences")
            except Exception as e:
                print(f"     Could not compare content: {e}")

            # 10. Save detailed comparison report
            comparison_report = output_dir / f"dataframe_comparison_report_{analysis_date_string}.txt"
            with open(comparison_report, 'w') as f:
                f.write("=== DATAFRAME COMPARISON REPORT ===\n\n")
                f.write(f"Old method rows: {len(final_results_old_method)}\n")
                f.write(f"Parquet method rows: {len(final_results_from_parquet)}\n\n")

                f.write("Columns comparison:\n")
                f.write(f"Old: {old_cols}\n")
                f.write(f"Parquet: {parquet_cols}\n\n")

                f.write("Data types:\n")
                for col in set(old_cols) & set(parquet_cols):
                    f.write(
                        f"{col}: Old={final_results_old_method[col].dtype}, Parquet={final_results_from_parquet[col].dtype}\n")

                f.write("\nNull value counts:\n")
                for col in set(old_cols) & set(parquet_cols):
                    old_null = final_results_old_method[col].isnull().sum()
                    parquet_null = final_results_from_parquet[col].isnull().sum()
                    if old_null != parquet_null:
                        f.write(f"{col}: Old={old_null}, Parquet={parquet_null}\n")

                f.write(f"\nDuplicate rows: Old={old_duplicates}, Parquet={parquet_duplicates}\n")
                f.write(f"\nMemory usage: Old={old_memory:.2f} MB, Parquet={parquet_memory:.2f} MB\n")

            print(f"\n📄 Detailed comparison report saved to: {comparison_report}")

            # Try to identify if the issue is with parquet file writing/reading
            print("\n10. Verifying parquet file integrity:")
            for pf in parquet_files[:3]:  # Check first 3 files
                test_read = pd.read_parquet(pf)
                print(f"   {pf.name}: {len(test_read)} rows, columns={list(test_read.columns)}")
                # Check if this matches the corresponding chunk in results
                chunk_idx = int(pf.stem.split('_')[-1]) - 1
                if chunk_idx < len(results):
                    if len(test_read) != len(results[chunk_idx]):
                        print(f"      ⚠️ Size mismatch: parquet={len(test_read)}, memory={len(results[chunk_idx])}")
                    elif not test_read.equals(results[chunk_idx]):
                        print(f"      ⚠️ Content mismatch detected in chunk {chunk_idx + 1}")
                    else:
                        print(f"      ✅ Chunk {chunk_idx + 1} matches")

            print("\n=== END OF COMPARISON ===\n")

        # Free memory from old method results
        del results
        del final_results_old_method
        import gc
        gc.collect()

        return final_results_from_parquet if not final_results_from_parquet.empty else pd.DataFrame()
    else:
        print("⚠ No results generated from old method")
        return final_results_from_parquet


def main():
    env_path = None
    if len(sys.argv) > 1:
        # Custom .env file path provided as command line argument
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        # Default to .env file in current directory
        load_dotenv()
        logger.info("Loading environment from default .env file")

    output_dir = os.environ['output_dir']
    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID
    dynamic_world_dir = os.environ['dynamic_world_dir']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))

    if not all_dynamic_world_files:
        logger.error(f"No .nc files found in {dynamic_world_dir}")
        sys.exit(1)

    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
    dynamic_world_data_file = os.environ['dynamic_world_data_file']
    download_recent_data = os.environ.get('download_recent_data', 'false').lower() == 'true'
    vector_lake_file = os.environ['vector_lake_file']
    new_dynamic_world_data_dir = os.environ['new_dynamic_world_data_dir']

    # Get analysis_date from environment
    analysis_date = os.environ.get('analysis_date', None)
    data_aggregation_period = os.environ.get('data_aggregation_period', 'monthly')

    # Get chunk size from environment or use default
    lake_chunk_size = int(os.environ.get('lake_chunk_size', '500'))
    print(f"Using lake chunk size: {lake_chunk_size}")

    if download_recent_data:
        logger.info("Downloading new dynamic world data...")
        new_dynamic_world_dataset_file = download_new_dynamic_world_data.download_new_dynamic_world_data(
            env_path=env_path)
        logger.debug(f"New dynamic world dataset file is: {new_dynamic_world_dataset_file}")
        logger.debug(f"Run near real time analysis for {new_dynamic_world_dataset_file}")

        results = precompute_nrt_breakpoints(
            input_nc_file=new_dynamic_world_dataset_file,
            output_dir=output_dir,
            lake_chunk_size=lake_chunk_size,
            analysis_date=analysis_date,
            data_aggregation_period=data_aggregation_period,
        )
        logger.debug(f"Results saved to {output_dir}")
    else:
        logger.debug(f"Not downloading new dynamic world data")
        logger.debug(f"Using {most_recent_dynamic_world_file} as input data.")
        file_size_gb = os.path.getsize(most_recent_dynamic_world_file) / (1024 ** 3)
        logger.info(f"Input file size: {file_size_gb:.2f} GB")

        results = precompute_nrt_breakpoints(
            input_nc_file=most_recent_dynamic_world_file,
            output_dir=output_dir,
            lake_chunk_size=lake_chunk_size,
            analysis_date=analysis_date,
            data_aggregation_period=data_aggregation_period,
        )
        logger.debug(f"Results saved to {output_dir}")

    print("\n✅ Near real-time breakpoint analysis completed successfully!")


if __name__ == "__main__":
    main()