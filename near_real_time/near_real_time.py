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
import psutil
import gc
# import os
# os.environ["OMP_NUM_THREADS"] = "8"  # Prevent thread oversubscription
# os.environ["MKL_NUM_THREADS"] = "8"
# os.environ["OPENBLAS_NUM_THREADS"] = "8"
# os.environ["NUMEXPR_NUM_THREADS"] = "8"


def log_memory_usage(stage="", threshold_mb=None):
    """Log current memory usage and optionally warn if above threshold"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    memory_gb = memory_mb / 1024

    log_msg = f"[MEMORY] {stage}: {memory_mb:.1f} MB ({memory_gb:.2f} GB)"
    print(log_msg)

    if threshold_mb and memory_mb > threshold_mb:
        print(f"⚠️ MEMORY WARNING: Exceeded threshold of {threshold_mb:.0f} MB")

    return memory_mb


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


def precompute_nrt_breakpoints(
        input_nc_file: str | Path,
        output_dir: str | Path,
        lake_chunk_size: int = 500,
        analysis_date: str | pd.Timestamp | None = None,
        data_aggregation_period: str = "monthly"
) -> pd.DataFrame:
    log_memory_usage("Start of function", threshold_mb=25000)  # Warn at 25GB

    input_nc_file = Path(input_nc_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # FIRST: Get lake IDs without loading full dataset
    print("Reading lake IDs from file...")
    with xr.open_dataset(input_nc_file, engine='netcdf4') as ds_meta:
        all_lake_ids = ds_meta.id_geohash.values
        total_lakes = len(all_lake_ids)
        print(f"Total lakes: {total_lakes}")

        log_memory_usage("After reading lake IDs")

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

    results = []  # Keep this for comparison - COMMENTED OUT TO SAVE MEMORY
    # Commenting out results list to save memory since we're not using old method
    # results = []

    # Process each chunk
    total_chunks = (total_lakes + lake_chunk_size - 1) // lake_chunk_size

    for chunk_idx, i in enumerate(range(0, total_lakes, lake_chunk_size)):
        chunk_num = chunk_idx + 1
        chunk_ids = all_lake_ids[i:i + lake_chunk_size]
        print(f"\nProcessing chunk {chunk_num}/{total_chunks}")

        log_memory_usage(f"Before chunk {chunk_num} loading", threshold_mb=28000)

        # Force garbage collection before loading new chunk
        gc.collect()

        # Load chunk with smaller chunk size for memory efficiency
        load_chunk_size = lake_chunk_size  # Don't load more than 100 lakes at once
        ds = xr.open_dataset(
            input_nc_file,
            engine='netcdf4',
            chunks={'id_geohash': load_chunk_size, 'date': -1}
        )
        ds_chunk = ds.sel(id_geohash=chunk_ids).load()
        ds.close()

        log_memory_usage(f"After loading chunk {chunk_num}")

        # Verify the chunk has data for analysis_date
        if analysis_date not in ds_chunk.date.values:
            print(f"  Warning: analysis_date {analysis_date} not in this chunk's dates")
            print(f"  Chunk date range: {ds_chunk.date.values[0]} to {ds_chunk.date.values[-1]}")
            # Use the most recent date from this chunk
            chunk_analysis_date = pd.to_datetime(ds_chunk.date.values[-1])
            print(f"  Using chunk's most recent date: {chunk_analysis_date}")
        else:
            chunk_analysis_date = analysis_date

        if 'date' in ds_chunk.coords:
            # Convert to pandas datetime if needed
            dates = pd.to_datetime(ds_chunk.date.values)
            # Reassign the date coordinate with proper datetime
            ds_chunk = ds_chunk.assign_coords(date=dates)
            print(f"  Fixed date coordinate for chunk {chunk_num}")

        # Check if chunk has valid lakes for this date
        ds_analysis_test = ds_chunk.sel(date=chunk_analysis_date)
        valid_lakes = ds_analysis_test.dropna(dim="id_geohash", how="all").id_geohash.values

        if len(valid_lakes) == 0:
            print(f"  No valid lakes for date {chunk_analysis_date}, skipping chunk")
            ds_chunk.close()
            del ds_chunk
            gc.collect()
            continue

        print(f"  Found {len(valid_lakes)} valid lakes for date {chunk_analysis_date}")

        # Create dataset wrapper for just this chunk
        dw_dataset_chunk = DWDataset(ds_chunk)

        # Initialize NRT breakpoint detector
        nrt_breakpoint = NRTBreakpoint(kwargs_break={})

        # Calculate breakpoints for this chunk
        chunk_output_file = chunk_output_dir / f"nrt_results_chunk_{chunk_num}_{total_chunks}.parquet"
        chunk_output_file_exists = os.path.exists(chunk_output_file)
        logger.debug(f"  Chunk output file {chunk_output_file} exists: {chunk_output_file_exists}")

        if not chunk_output_file_exists:
            try:
                log_memory_usage(f"Before calculate_break for chunk {chunk_num}")

                chunk_result = nrt_breakpoint.calculate_break(
                    dataset=dw_dataset_chunk,
                    analysis_date=chunk_analysis_date,
                    data_aggregation_period=data_aggregation_period,
                    object_id=chunk_ids,
                )

                log_memory_usage(f"After calculate_break for chunk {chunk_num}")

                if chunk_result is not None and len(chunk_result) > 0:
                    # Save chunk results
                    chunk_result.to_parquet(chunk_output_file, index=True)
                    print(f"  Saved chunk results to {chunk_output_file} ({len(chunk_result)} rows)")

                    # Optional: Keep for comparison (commented out to save memory)
                    # results.append(chunk_result)

                    # Clear chunk_result immediately
                    del chunk_result
                else:
                    print(f"  No results generated for chunk")

            except ValueError as e:
                if "n_jobs == 0" in str(e):
                    print(f"  No valid lakes for analysis in this chunk")
                else:
                    print(f"  Error processing chunk: {e}")
                continue
            except MemoryError as e:
                print(f"  MEMORY ERROR in chunk {chunk_num}: {e}")
                print(f"  Consider reducing lake_chunk_size further")
                raise

        # Aggressive memory cleanup
        ds_chunk.close()
        del ds_chunk
        del dw_dataset_chunk
        if 'ds_analysis_test' in locals():
            del ds_analysis_test
        if 'nrt_breakpoint' in locals():
            del nrt_breakpoint

        # Force garbage collection
        gc.collect()

        log_memory_usage(f"After cleanup for chunk {chunk_num}", threshold_mb=28000)

    # Combine results from parquet files using Dask (memory-efficient)
    final_output_file_name_from_parquet = "nrt_breakpoints_all_lakes_from_parquet_" + analysis_date_string + ".parquet"
    final_output_file_from_parquet = output_dir / final_output_file_name_from_parquet

    parquet_files = sorted(chunk_output_dir.glob("nrt_results_chunk_*.parquet"))

    if parquet_files:
        print(f"\n📂 Found {len(parquet_files)} chunk parquet files to combine")
        log_memory_usage("Before Dask read")

        # Use Dask to read all parquet files lazily
        print("  Using Dask for out-of-core processing...")
        ddf = dd.read_parquet(
            str(chunk_output_dir / "nrt_results_chunk_*.parquet"),
            engine='pyarrow'
        )

        print(f"  Dask DataFrame partitions: {ddf.npartitions}")
        print(f"  Dask DataFrame columns: {list(ddf.columns)}")

        # Compute the result
        print("  Computing combined DataFrame (this may take a moment)...")
        final_results_from_parquet = ddf.compute()

        log_memory_usage("After Dask compute")

        # Save combined results
        final_results_from_parquet.to_parquet(final_output_file_from_parquet, index=True)
        print(f"\n✅ Combined results from parquet files saved to {final_output_file_from_parquet}")
        print(f"   Total rows: {len(final_results_from_parquet):,}")
        print(f"   Memory usage: {final_results_from_parquet.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    else:
        print("⚠ No parquet files found to combine")
        final_results_from_parquet = pd.DataFrame()

    # OLD METHOD IS DISABLED TO SAVE MEMORY
    # We're not keeping results in memory anymore, so just return the parquet results
    print("\n✅ Using Dask/parquet method (memory-optimized)")

    # Final cleanup
    gc.collect()
    log_memory_usage("End of function")

    return final_results_from_parquet


def main():
    env_path = None
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
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

    analysis_date = os.environ.get('analysis_date', None)
    data_aggregation_period = os.environ.get('data_aggregation_period', 'monthly')
    lake_chunk_size = int(os.environ.get('lake_chunk_size', '200'))  # Reduced default from 500 to 200
    print(f"🔧 LAKE_CHUNK_SIZE from environment: {lake_chunk_size}")
    print(f"🔧 All relevant env vars:")
    for var in ['lake_chunk_size', 'data_aggregation_period', 'analysis_date']:
        print(f"   {var} = {os.environ.get(var, 'NOT SET')}")
    print(f"Using lake chunk size: {lake_chunk_size}")
    log_memory_usage("Main start")

    # Safely get file size without causing memory issues
    try:
        file_size_bytes = os.path.getsize(most_recent_dynamic_world_file)
        file_size_gb = file_size_bytes / (1024 ** 3)
        logger.info(f"Input file size: {file_size_gb:.2f} GB")
    except (OSError, IOError) as e:
        logger.warning(f"Could not get file size: {e}")
        file_size_gb = None

    if download_recent_data:
        logger.info("Downloading new dynamic world data...")
        new_dynamic_world_dataset_file = download_new_dynamic_world_data.download_new_dynamic_world_data_split_files_v1(
            env_path=env_path)
        logger.debug(f"New dynamic world dataset file is: {new_dynamic_world_dataset_file}")

        results = precompute_nrt_breakpoints(
            input_nc_file=new_dynamic_world_dataset_file,
            output_dir=output_dir,
            lake_chunk_size=lake_chunk_size,
            analysis_date=analysis_date,
            data_aggregation_period=data_aggregation_period,
        )
    else:
        logger.debug(f"Using {most_recent_dynamic_world_file} as input data.")

        results = precompute_nrt_breakpoints(
            input_nc_file=most_recent_dynamic_world_file,
            output_dir=output_dir,
            lake_chunk_size=lake_chunk_size,
            analysis_date=analysis_date,
            data_aggregation_period=data_aggregation_period,
        )

    log_memory_usage("Main end")
    print("\n✅ Near real-time breakpoint analysis completed successfully!")


if __name__ == "__main__":
    main()