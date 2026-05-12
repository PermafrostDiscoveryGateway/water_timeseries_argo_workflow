import sys
from dotenv import load_dotenv
import os
import datetime
from loguru import logger
from water_timeseries.downloader import EarthEngineDownloader
from google.cloud import storage
from pathlib import Path
import time

# Load environment variables from .env file (optional custom path)
if len(sys.argv) > 1:
    # Custom .env file path provided as command line argument
    env_path = sys.argv[1]
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loading environment from: {env_path}")
else:
    # Default to .env file in current directory
    load_dotenv()
    logger.info("Loading environment from default .env file")

project = os.environ['project']
base_dir = os.environ['base_dir']
dynamic_world_dir = os.environ['dynamic_world_dir']
year = os.environ['year']
split_vector_dataset_dir = os.environ['split_vector_dataset_dir']
start_split_vector_num= int(os.environ['start_split_vector_num'])
end_split_vector_num= int(os.environ['end_split_vector_num'])

existing_dynamic_world_files = os.listdir(dynamic_world_dir)

split_vector_nums = list(range(start_split_vector_num, end_split_vector_num+1))

logger.debug(f"Getting all the split vector files")
all_split_vector_files = os.listdir(split_vector_dataset_dir)
logger.debug(f"We have {len(all_split_vector_files)} split vector files")


EE_PROJECT_ID = project
os.environ["EE_PROJECT"] = EE_PROJECT_ID
client = storage.Client(project=project)

current_datetime = datetime.datetime.now()
timestamp = str(current_datetime).replace(' ', '_').replace(':', '-')  # Fixed the replace issue
current_year = current_datetime.year

all_years = list(range(2015, current_year))
all_months = list(range(6, 11))

dynamic_world_dataset_name = 'dynamic_world_' + timestamp + '.zarr'
dynamic_world_dataset_path = os.path.join(dynamic_world_dir, dynamic_world_dataset_name)

dl = EarthEngineDownloader(ee_auth=True, logger=logger)

for split_vector_file in all_split_vector_files:
    logger.debug(f"Downloading all years for {split_vector_file}")
    time.sleep(5)
    path_to_split_vector_file = os.path.join(split_vector_dataset_dir, split_vector_file)
    chunk_num = str(int(Path(path_to_split_vector_file).stem.split('_')[-1]))
    output_filename = 'dynamic_world_split_' + chunk_num + '.zarr'
    output_filepath = os.path.join(dynamic_world_dir, output_filename)
    if os.path.exists(output_filepath):
        logger.debug(f"File {output_filepath} already exists, skipping")
    else:
        logger.debug(f"Downloading to {output_filepath}")
        dl = EarthEngineDownloader(ee_auth=True, logger=logger)
        ds = dl.download_dw_monthly(
            vector_dataset=path_to_split_vector_file,
            name_attribute="id_geohash",
            years=all_years,
            months=all_months,
            save_to_file=output_filepath,
            max_total_requests=2000,
            n_parallel=6,
        )
        logger.debug(f"Finished downloading all years for {split_vector_file}")

logger.debug(f"Finished downloading all years for {dynamic_world_dataset_name}")
