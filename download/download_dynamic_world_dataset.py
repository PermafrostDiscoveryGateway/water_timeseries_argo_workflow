from dotenv import load_dotenv
import os
from google.cloud import storage

# Load environment variables from .env file
load_dotenv()
print("downloading the file")
project = os.environ['project']
historical_dynamic_world_file= os.environ['historical_dynamic_world_file']
base_dir= os.environ['base_dir']
dynamic_world_historical_dir  = os.environ['dynamic_world_historical_dir']

EE_PROJECT_ID = project

client = storage.Client(project=project)

gs_url = "gs://"+historical_dynamic_world_file
# Remove 'gs://' prefix
bucket_name = gs_url.replace('gs://', '').split('/')[0]
blob_path = '/'.join(gs_url.replace('gs://', '').split('/')[1:])

bucket = client.bucket(bucket_name)
blob = bucket.blob(blob_path)
blob_name = blob.name
blob_name_stem = blob_name.split('/')[-1]
local_path = os.path.join(base_dir, 'input', blob_name_stem )
# Download to local file
blob.download_to_filename(local_path)
print("Downloaded the data")

