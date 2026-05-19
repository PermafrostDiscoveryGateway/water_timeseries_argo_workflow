from loguru import logger
import os
import glob
import sys
from dotenv import load_dotenv
import download_new_dynamic_world_data
from water_timeseries.breakpoint import NRTBreakpoint
from water_timeseries.dataset import DWDataset
import xarray as xr
import pandas as pd
from pathlib import Path


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

        if analysis_date is None:
            dates = ds_meta.date.values
            analysis_date = pd.to_datetime(dates[-1])
            print(f"Using most recent date: {analysis_date}")

    results = []

    # Process each chunk by loading ONLY that chunk of data
    for i in range(0, total_lakes, lake_chunk_size):
        chunk_ids = all_lake_ids[i:i + lake_chunk_size]
        print(f"\nProcessing chunk {i // lake_chunk_size + 1}/{(total_lakes + lake_chunk_size - 1) // lake_chunk_size} "
              f"({len(chunk_ids)} lakes)...")

        # Load only this chunk using Dask
        ds = xr.open_dataset(
            input_nc_file,
            engine='netcdf4',
            chunks={'id_geohash': lake_chunk_size, 'date': -1}
        )
        ds_chunk = ds.sel(id_geohash=chunk_ids).load()
        ds.close()

        # Create dataset wrapper for just this chunk
        dw_dataset_chunk = DWDataset(ds_chunk)

        # Initialize NRT breakpoint detector
        nrt_breakpoint = NRTBreakpoint(kwargs_break={})

        # Calculate breakpoints for this chunk
        chunk_result = nrt_breakpoint.calculate_break(
            dataset=dw_dataset_chunk,
            analysis_date=analysis_date,
            data_aggregation_period=data_aggregation_period,
            object_id=str(chunk_ids),
        )

        results.append(chunk_result)

        # Save chunk results as Parquet
        chunk_output_file = output_dir / f"nrt_results_chunk_{i // lake_chunk_size + 1}.parquet"
        chunk_result.to_parquet(chunk_output_file, index=True)
        print(f"  Saved chunk results to {chunk_output_file}")

        # Clear memory
        ds_chunk.close()
        del ds_chunk
        del dw_dataset_chunk
        import gc
        gc.collect()

        print(f"  Memory cleared for next chunk")

    # Combine all results
    if results:
        final_results = pd.concat(results, axis=0)

        # Save final results as Parquet
        final_output_file = output_dir / "nrt_breakpoints_all_lakes.parquet"
        final_results.to_parquet(final_output_file, index=True)
        print(f"\n✅ Final results saved to {final_output_file}")

        # Print summary statistics
        print(f"\n📊 Summary Statistics:")
        print(f"  - Total lakes processed: {len(final_results)}")
        if 'water_predicted' in final_results.columns:
            print(f"  - Lakes with predictions: {final_results['water_predicted'].notna().sum()}")
            print(f"  - Mean predicted water: {final_results['water_predicted'].mean():.4f}")
            print(f"  - Mean residual: {final_results['water_residual'].mean():.4f}")

        # Print file size
        file_size_gb = final_output_file.stat().st_size / (1024 ** 3)
        print(f"  - Output file size: {file_size_gb:.2f} GB")

        return final_results
    else:
        print("⚠ No results generated")
        return pd.DataFrame()


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