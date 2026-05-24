import netCDF4 as nc
import pandas as pd
from netCDF4 import num2date
from datetime import datetime
from loguru import logger
import os
import numpy as np
import glob
import xarray as xr
from dotenv import load_dotenv

def combine_new_dynamic_world_data_with_latest(env_path=None):
    # Load environment with fallback logic
    if env_path is None:
        load_dotenv()
        logger.info("Loading environment from default .env file")
    else:
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")

    # Get environment variables (now guaranteed to exist after validation)
    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID
    dynamic_world_dir = os.environ['dynamic_world_dir']
    logger.debug(f"dynamic_world_dir: {dynamic_world_dir}")
    split_new_dynamic_world_data_dir = os.environ['split_new_dynamic_world_data_dir']
    logger.debug(f"split_new_dynamic_world_data_dir: {split_new_dynamic_world_data_dir}")

    split_contents = os.listdir(split_new_dynamic_world_data_dir)
    logger.debug(f"Split contents: {split_contents}")