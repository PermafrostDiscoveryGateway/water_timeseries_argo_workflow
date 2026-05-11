from dotenv import load_dotenv
import os
import datetime
from loguru import logger
from water_timeseries.downloader import EarthEngineDownloader
from google.cloud import storage
import ee
# Load environment variables from .env file
load_dotenv()

bbox_alaska = {
    'west': -179.0,   # or -179.9 for western Aleutians
    'south': 51.0,    # southern panhandle tip
    'east': -130.0,   # eastern panhandle
    'north': 71.5     # northern coast
}

bbox_ca_1 = {
    'west': -141.0,   # Yukon/Alaska border
    'south': 41.7,
    'east': -96.8,    # ~ central Manitoba/Saskatchewan border
    'north': 83.1
}

bbox_ca_2 = {
    'west': -96.8,
    'south': 41.7,
    'east': -52.6,    # Newfoundland
    'north': 83.1
}

bbox_eu_1 = {
    'west': -30.0,    # Including Iceland, Svalbard
    'south': 51.0,
    'east': 11.4,
    'north': 85.0
}

bbox_eu_2 = {
    'west': 11.4,
    'south': 51.0,
    'east': 52.8,
    'north': 85.0
}

bbox_eu_3 = {
    'west': 52.8,
    'south': 51.0,
    'east': 94.2,
    'north': 85.0
}


bbox_eu_4 = {
    'west': 94.2,
    'south': 51.0,
    'east': 135.6,
    'north': 85.0
}

bbox_eu_5 = {
    'west': 135.6,
    'south': 51.0,
    'east': 177.0,    # Your original eastern limit
    'north': 85.0
}

project = os.environ['project']
base_dir= os.environ['base_dir']
dynamic_world_dir  = os.environ['dynamic_world_dir']
year = os.environ['year']
bbox_west = os.environ.get('bbox_west')
bbox_east = os.environ.get('bbox_east')
bbox_north = os.environ.get('bbox_north')
bbox_south = os.environ.get('bbox_south')
if bbox_west and bbox_east and bbox_north and bbox_south:
    bbox_west = float(bbox_west)
    bbox_east = float(bbox_east)
    bbox_south = float(bbox_south)
    bbox_north = float(bbox_north)

if ',' in year:
    year_parts = year.split(',')
    new_years = []
    for part in year_parts:
        new_year = int(part)
        new_years.append(new_year)
    year = new_years
else:
    year = [int(year)]
vector_dataset_path = os.path.join(base_dir, 'input', 'Nitze_etal_Lakes_filtered_full_set_V2d.parquet')

EE_PROJECT_ID = project
os.environ["EE_PROJECT"] = EE_PROJECT_ID
client = storage.Client(project=project)
try:
    ee.Initialize(project=EE_PROJECT_ID)
    # Test a simple operation
    test_image = ee.Image('LANDSAT/LC08/C02/T1_TOA/LC08_044034_20140318')
    test_info = test_image.getInfo()
    logger.info("GEE initialized successfully")
except Exception as e:
    logger.error(f"GEE initialization failed: {e}")
    raise

current_datetime = datetime.datetime.now()
timestamp = str(current_datetime.replace).replace(' ', '_')
current_year = current_datetime.year

all_years = list(range(2015, current_year ))
all_months = list(range(6, 11))

dynamic_world_dataset_name = 'dynamic_world_' + timestamp + '.zarr'
dynamic_world_dataset_path = os.path.join(dynamic_world_dir, dynamic_world_dataset_name)

if year == [2017, 2018]:
    print('same')

dl = EarthEngineDownloader(ee_auth=True, logger=logger)

if bbox_north and bbox_east and bbox_west and bbox_south:
    logger.debug(f"We have a bounding box")

    ds = dl.download_dw_monthly(
        vector_dataset=vector_dataset_path,
        name_attribute="id_geohash",
        years=year,
        months=all_months,
        save_to_file=dynamic_world_dataset_path,
        bbox_north=bbox_north,
        bbox_east=bbox_east,
        bbox_west=bbox_west,
        bbox_south=bbox_south,
        max_total_requests=100,  # Reduce from 500 to 100
        n_parallel=1,
    )


    if ds is None or len(ds.coords['id_geohash']) == 0:
        logger.error(f"No data downloaded for year {year}")
        # Try with no_download mode to see what would be requested
        dl.download_dw_monthly(
            vector_dataset=vector_dataset_path,
            name_attribute="id_geohash",
            years=year,
            months=all_months,
            bbox_north=bbox_north,
            bbox_east=bbox_east,
            bbox_west=bbox_west,
            bbox_south=bbox_south,
            no_download=True,  # Just logs what would be downloaded
        )
        raise ValueError(f"Empty dataset for year {year}")
else:
    logger.debug(f"We do not have a bounding box")

    ds = dl.download_dw_monthly(
        vector_dataset=vector_dataset_path,
        name_attribute="id_geohash",
        years=year,
        months=all_months,
        save_to_file=dynamic_world_dataset_path,
        max_total_requests=100,  # Reduce from 500 to 100
        n_parallel=1,
    )

    if ds is None or len(ds.coords['id_geohash']) == 0:
        logger.error(f"No data downloaded for year {year}")
        # Try with no_download mode to see what would be requested
        dl.download_dw_monthly(
            vector_dataset=vector_dataset_path,
            name_attribute="id_geohash",
            years=year,
            months=all_months,
            no_download=True,  # Just logs what would be downloaded
        )
        raise ValueError(f"Empty dataset for year {year}")



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

