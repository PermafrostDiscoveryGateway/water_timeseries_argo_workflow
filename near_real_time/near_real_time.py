import netCDF4 as nc
import pandas as pd
from netCDF4 import num2date
from datetime import datetime
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


def load_environment():
    """
    Load environment variables with fallback priority:
    1. Command line argument (.env file path)
    2. Default ./.env file
    3. Kubernetes/OS environment variables (already present)
    """
    env_path = None

    # Priority 1: Command line argument for .env file
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        if Path(env_path).exists():
            load_dotenv(dotenv_path=env_path, override=False)  # Don't override existing env vars
            logger.info(f"Loaded environment from command line .env: {env_path}")
        else:
            logger.warning(f".env file not found at {env_path}, checking other sources")

    # Priority 2: Default .env file in current directory
    if not env_path or not Path(env_path).exists():
        default_env = Path.cwd() / ".env"
        if default_env.exists():
            load_dotenv(dotenv_path=default_env, override=False)
            logger.info(f"Loaded environment from default .env: {default_env}")
        else:
            logger.info("No .env file found, using Kubernetes/OS environment variables")

    # Priority 3: Kubernetes/OS environment variables are already in os.environ

    # Validate required variables (with helpful error messages)
    required_vars = [
        'output_dir',
        'project',
        'dynamic_world_dir',
        'vector_lake_file',
        'new_dynamic_world_data_dir'
    ]

    missing_vars = []
    for var in required_vars:
        if var not in os.environ:
            missing_vars.append(var)

    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        logger.info("Available environment variables: {list(os.environ.keys())}")
        raise EnvironmentError(error_msg)

    # Optional variables with defaults
    if 'dynamic_world_data_file' not in os.environ:
        logger.warning("dynamic_world_data_file not set, will use most recent file")

    # Log which source is providing each variable (debug)
    logger.debug("Environment configuration:")
    for var in required_vars:
        source = "K8s/OS" if var not in locals() else ".env"
        logger.debug(f"  {var} = {os.environ[var]} (source: {source})")


def precompute_nrt_breakpoints(
        input_nc_file: str | Path,
        output_dir: str | Path,
        lake_chunk_size: int = 2000,
        n_jobs: int = 1,
        analysis_date: str | pd.Timestamp | None = None,
        data_aggregation_period: str = "monthly"
) -> pd.DataFrame:
    """[Your existing function remains unchanged]"""

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
            object_id=chunk_ids
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
    # Load environment with fallback logic
    load_environment()

    # Now all variables should be available in os.environ
    output_dir = os.environ['output_dir']
    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID
    dynamic_world_dir = os.environ['dynamic_world_dir']

    # Handle optional variable with fallback logic
    dynamic_world_data_file = os.environ.get('dynamic_world_data_file')
    if dynamic_world_data_file:
        logger.info(f"Using specified dynamic world data file: {dynamic_world_data_file}")
    else:
        all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
        if all_dynamic_world_files:
            dynamic_world_data_file = max(all_dynamic_world_files, key=os.path.getctime)
            logger.info(f"No dynamic_world_data_file specified, using most recent: {dynamic_world_data_file}")
        else:
            logger.error(f"No .nc files found in {dynamic_world_dir}")
            sys.exit(1)

    vector_lake_file = os.environ['vector_lake_file']
    new_dynamic_world_data_dir = os.environ['new_dynamic_world_data_dir']

    # Get env_path for download function (if provided via command line)
    env_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Download new data
    new_dynamic_world_dataset_file = download_new_dynamic_world_data.download_new_dynamic_world_data_split_files(env_path=env_path)
    logger.debug(f"New dynamic world dataset file is: {new_dynamic_world_dataset_file}")
    logger.debug(f"Run near real time analysis for {new_dynamic_world_dataset_file}")

    # Run analysis
    results = precompute_nrt_breakpoints(
        input_nc_file=new_dynamic_world_dataset_file,
        output_dir=output_dir,
        lake_chunk_size=2000,
        n_jobs=1
    )
    logger.debug(f"Results saved to {output_dir} : {results}")


if __name__ == "__main__":
    main()