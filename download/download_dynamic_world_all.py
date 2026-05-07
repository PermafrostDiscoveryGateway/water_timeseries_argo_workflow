from dotenv import load_dotenv
import os
import datetime
from loguru import logger
from water_timeseries.downloader import EarthEngineDownloader
from google.cloud import storage

# Load environment variables from .env file
load_dotenv()

project = os.environ['project']
base_dir= os.environ['base_dir']
dynamic_world_dir  = os.environ['dynamic_world_dir']
vector_dataset_path = os.path.join(base_dir, 'input', 'Nitze_etal_Lakes_filtered_full_set_V2d.parquet')

EE_PROJECT_ID = project
os.environ["EE_PROJECT"] = EE_PROJECT_ID
client = storage.Client(project=project)

current_datetime = datetime.datetime.now()
timestamp = str(current_datetime.replace).replace(' ', '_')
current_year = current_datetime.year

all_years = list(range(2015, current_year + 1))
all_months = list(range(1, 13))

dynamic_world_dataset_name = 'dynamic_world_' + timestamp + '.zarr'
dynamic_world_dataset_path = os.path.join(dynamic_world_dir, dynamic_world_dataset_name)

dl = EarthEngineDownloader(ee_auth=True, logger=logger)
ds = dl.download_dw_monthly(
    vector_dataset=vector_dataset_path,
    name_attribute="id_geohash",
    years=all_years,
    months=all_months,
    save_to_file=dynamic_world_dataset_path,
)

print('finished downloading')
