import os
import toml
from loguru import logger
from water_timeseries.downloader import EarthEngineDownloader
import nest_asyncio
import sys
import asyncio
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
    VECTOR_DATASET = config["VECTOR_DATASET"]
    DOWNLOAD_DIR = config["DOWNLOAD_DIR"]
    START_YEAR = config["START_YEAR"]
    END_YEAR = config["END_YEAR"]
    START_MONTH = config["START_MONTH"]
    END_MONTH = config["END_MONTH"]
    FORMAT = 'nc'
    APPEND = config["APPEND"]
    ENTIRE = config["ENTIRE"]
    # if not append, download file for the given range
    if ENTIRE:
        print(f"Download all data from start to present day in a new file")
    elif APPEND:
        print(f"Append new data to most recent, complete file")
    else:
        print(f"Download all data from start to the end")
        MONTHS = list(range(START_MONTH, END_MONTH + 1))
        YEARS = list(range(START_YEAR, END_YEAR + 1))
        print('months and years')


    print("Done")
if __name__ == "__main__":
    main()