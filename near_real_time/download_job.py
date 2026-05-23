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


    if download_recent_data:
        logger.info("Downloading new dynamic world data...")
        new_dynamic_world_dataset_file = download_new_dynamic_world_data.download_new_dynamic_world_data(
            env_path=env_path)
        logger.debug(f"New dynamic world dataset file is: {new_dynamic_world_dataset_file}")


if __name__ == "__main__":
    main()