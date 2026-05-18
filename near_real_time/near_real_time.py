from loguru import logger
import os
import glob
import sys
from dotenv import load_dotenv
from . import download_new_dynamic_world_data
from water_timeseries.breakpoint import NRTBreakpoint
from water_timeseries.dataset import DWDataset
import xarray as xr
import pandas as pd
from pathlib import Path


def precompute_nrt_breakpoints(
        input_nc_file: str | Path,
        output_dir: str | Path,
        lake_chunk_size: int = 2000,
        analysis_date: str | pd.Timestamp | None = None,
        data_aggregation_period: str = "monthly"
) -> pd.DataFrame:
    """
    Precompute NRT breakpoints for a Dynamic World dataset.

    This replicates the functionality of:
    uv run water-timeseries nrt-precompute \
        downloads/lakes_dw_V2d.nc \
        --output-dir precomputed/nrt \
        --lake-chunk-size 2000 \
        --n-jobs 1

    Parameters
    ----------
    input_nc_file : str or Path
        Path to the Dynamic World NetCDF file
    output_dir : str or Path
        Directory where output files will be saved
    lake_chunk_size : int, optional
        Number of lakes to process in each chunk (default: 2000)
    n_jobs : int, optional
        Number of parallel jobs for processing (default: 1)
    analysis_date : str or pd.Timestamp, optional
        Date for NRT analysis. If None, uses the most recent date in the dataset
    data_aggregation_period : str, optional
        Period for data aggregation, either "all" or "monthly" (default: "monthly")

    Returns
    -------
    pd.DataFrame
        DataFrame containing breakpoint analysis results
    """

    # Convert to Path objects
    input_nc_file = Path(input_nc_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    print(f"Loading dataset from {input_nc_file}...")
    ds = xr.open_dataset(input_nc_file)
    dw_dataset = DWDataset(ds)

    # Determine analysis date if not provided
    if analysis_date is None:
        # Use the most recent date in the dataset
        analysis_date = dw_dataset.dates_[-1]
        print(f"No analysis date provided. Using most recent date: {analysis_date}")

    # Initialize NRT breakpoint detector
    nrt_breakpoint = NRTBreakpoint(kwargs_break={})

    # Get all lake IDs
    all_lake_ids = dw_dataset.object_ids_
    total_lakes = len(all_lake_ids)
    print(f"Processing {total_lakes} lakes...")

    # Process lakes in chunks
    results = []
    for i in range(0, total_lakes, lake_chunk_size):
        chunk_ids = all_lake_ids[i:i + lake_chunk_size]
        print(f"Processing chunk {i // lake_chunk_size + 1}/{(total_lakes + lake_chunk_size - 1) // lake_chunk_size} "
              f"({len(chunk_ids)} lakes)...")

        # Calculate breakpoints for this chunk
        chunk_result = nrt_breakpoint.calculate_break(
            dataset=dw_dataset,
            analysis_date=analysis_date,
            data_aggregation_period=data_aggregation_period,
            object_id=chunk_ids,
        )

        results.append(chunk_result)

        # Save intermediate results
        chunk_output_file = output_dir / f"nrt_results_chunk_{i // lake_chunk_size + 1}.csv"
        chunk_result.to_csv(chunk_output_file)
        print(f"  Saved chunk results to {chunk_output_file}")

    # Combine all results
    if results:
        final_results = pd.concat(results, axis=0)

        # Save final results
        final_output_file = output_dir / "nrt_breakpoints_all_lakes.csv"
        final_results.to_csv(final_output_file)
        print(f"\n✅ Final results saved to {final_output_file}")

        # Also save as parquet for faster reading (optional)
        parquet_output_file = output_dir / "nrt_breakpoints_all_lakes.parquet"
        final_results.to_parquet(parquet_output_file)
        print(f"✅ Parquet format saved to {parquet_output_file}")

        # Print summary statistics
        print(f"\n📊 Summary Statistics:")
        print(f"  - Total lakes processed: {len(final_results)}")
        print(f"  - Lakes with predictions: {final_results['water_predicted'].notna().sum()}")
        print(f"  - Mean predicted water: {final_results['water_predicted'].mean():.4f}")
        print(f"  - Mean residual: {final_results['water_residual'].mean():.4f}")

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

    # environment variables are read here so it works with a .env file, or if they are
    # in the yaml file for an argo workflow
    output_dir = os.environ['output_dir']
    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID
    dynamic_world_dir = os.environ['dynamic_world_dir']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
    dynamic_world_data_file = os.environ['dynamic_world_data_file']
    vector_lake_file = os.environ['vector_lake_file']
    new_dynamic_world_data_dir = os.environ['new_dynamic_world_data_dir']


    new_dynamic_world_dataset_file = download_new_dynamic_world_data.download_new_dynamic_world_data(env_path=env_path)
    logger.debug(f"New dynamic world dataset file is: {new_dynamic_world_dataset_file}")
    logger.debug(f"Run near real time analysis for {new_dynamic_world_dataset_file}")

    results = precompute_nrt_breakpoints(
        input_nc_file=new_dynamic_world_dataset_file,
        output_dir=output_dir,
        lake_chunk_size=2000,
    )
    logger.debug(f"Results saved to {output_dir} : {results}")

if __name__ == "__main__":
    main()


