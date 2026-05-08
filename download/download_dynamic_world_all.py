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

all_years = list(range(2015, current_year ))
all_months = list(range(6, 11))

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


# Your bucket path
bucket_path = "pdg-storage-default/water_timeseries_v2/data/input"

# Split into bucket name and prefix
bucket_name = bucket_path.split('/')[0]  # "pdg-storage-default"
blob_prefix = '/'.join(bucket_path.split('/')[1:])  # "water_timeseries_v2/data/input"

# Initialize client
bucket = client.bucket(bucket_name)

# Upload a file
local_file_path = dynamic_world_dataset_path
blob_path = f"{blob_prefix}/{dynamic_world_dataset_name}"  # Full path including filename

blob = bucket.blob(blob_path)
blob.upload_from_filename(local_file_path)

print(f"Uploaded to: gs://{bucket_name}/{blob_path}")

