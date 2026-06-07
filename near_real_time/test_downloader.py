#!/usr/bin/env python3
import sys
import os
from dotenv import load_dotenv

# Load environment
if len(sys.argv) > 1:
    load_dotenv(dotenv_path=sys.argv[1])
else:
    load_dotenv()

EE_PROJECT_ID = os.environ.get('EE_PROJECT', 'pdg-project-406720')
print(f"Using project: {EE_PROJECT_ID}")

# Import and test EarthEngineDownloader
from water_timeseries.downloader import EarthEngineDownloader

try:
    downloader = EarthEngineDownloader(ee_project=EE_PROJECT_ID)
    print("✓ EarthEngineDownloader initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    import traceback
    traceback.print_exc()