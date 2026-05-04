import os
import toml
from loguru import logger
from water_timeseries.downloader import EarthEngineDownloader
import nest_asyncio
import sys
from pathlib import Path
import asyncio
import datetime
# Set the event loop policy for better compatibility
if sys.platform == 'darwin':  # macOS
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
else:  # Linux (Kubernetes)
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Only use nest_asyncio if you're in a nested environment
try:
    loop = asyncio.get_running_loop()
    nest_asyncio.apply()  # Only if already running in a loop
except RuntimeError:
    pass

def human_file_size(path):
    size = path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"

def load_config(config_path="/app/config/config.toml"):
    """Load configuration from TOML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = toml.load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download Dynamic World data")
    parser.add_argument("--config", help="Path to config file", default="/app/config/config.toml")
    # Load TOML config
    args = parser.parse_args()
    config_path = "/app/config/config.toml"
    if args.config:
        config_path = args.config
    config = load_config(config_path=config_path)
    EE_PROJECT = config["EE_PROJECT"]
    os.environ["EE_PROJECT"] = EE_PROJECT
    VECTOR_DATASET = config["VECTOR_DATASET"]
    DOWNLOAD_DIR = config["DOWNLOAD_DIR"]
    START_YEAR = config["START_YEAR"]
    END_YEAR = config["END_YEAR"]
    START_MONTH = config["START_MONTH"]
    END_MONTH = config["END_MONTH"]
    FORMAT = config["FORMAT"]
    APPEND = config["APPEND"]
    ENTIRE = config["ENTIRE"]
    # if not append, download file for the given range
    current_datetime = datetime.datetime.now()
    current_year = current_datetime.year
    current_month = current_datetime.month
    CURRENT_TIMESTAMP = str(current_datetime).replace(" ", "_")
    NEW_FILENAME = 'dynamic_world_data_' +CURRENT_TIMESTAMP + '.' + FORMAT
    DOWNLOAD_FILEPATH = os.path.join(DOWNLOAD_DIR, NEW_FILENAME)
    if ENTIRE:
        print(f"Download all data from start to present day in a new file")
        start_year = 2015
        start_month = 1
        MONTHS = list(range(start_year, current_year + 1))
        YEARS = list(range(start_month, current_month + 1))
        dl = EarthEngineDownloader(ee_auth=True, logger=logger)
        ds = dl.download_dw_monthly(
            vector_dataset=VECTOR_DATASET,
            name_attribute="id_geohash",
            years=YEARS,
            months=MONTHS,
            save_to_file=str(DOWNLOAD_FILEPATH),
        )
        new_file_exists = os.path.exists(DOWNLOAD_FILEPATH)
        if new_file_exists:
            print(f"New file {DOWNLOAD_FILEPATH} exists")
            readable_filesize = human_file_size(Path(DOWNLOAD_FILEPATH))
            print(f"New file {DOWNLOAD_FILEPATH} size: {readable_filesize}")
    elif APPEND:
        print(f"Append new data to most recent, complete file")
        downloaded_files = os.listdir(DOWNLOAD_DIR)
        files = [os.path.join(DOWNLOAD_DIR, f) for f in os.listdir(DOWNLOAD_DIR)
                 if os.path.isfile(os.path.join(DOWNLOAD_DIR, f))]

        # Get the most recent (by modification time)
        most_recent = max(files, key=os.path.getmtime)
        print(f"Most recent file { most_recent }")
    else:
        print(f"Download all data from start to the end")
        MONTHS = list(range(START_MONTH, END_MONTH + 1))
        YEARS = list(range(START_YEAR, END_YEAR + 1))
        dl = EarthEngineDownloader(ee_auth=True, logger=logger)
        ds = dl.download_dw_monthly(
            vector_dataset=VECTOR_DATASET,
            name_attribute="id_geohash",
            years=YEARS,
            months=MONTHS,
            save_to_file=str(DOWNLOAD_FILEPATH),
        )
        print('months and years')
        new_file_exists = os.path.exists(DOWNLOAD_FILEPATH)
        if new_file_exists:
            print(f"New file {DOWNLOAD_FILEPATH} exists")
            readable_filesize = human_file_size(Path(DOWNLOAD_FILEPATH))
            print(f"New file {DOWNLOAD_FILEPATH} size: {readable_filesize}")


    print("Done")
if __name__ == "__main__":
    main()