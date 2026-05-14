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
    new_dynamic_world_dataset_file = download_new_dynamic_world_data.download_new_dynamic_world_data(env_path=env_path)
    logger.debug(f"New dynamic world dataset file is: {new_dynamic_world_dataset_file}")

if __name__ == "__main__":
    main()


