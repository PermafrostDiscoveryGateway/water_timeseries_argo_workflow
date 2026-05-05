import os
import toml
from loguru import logger
from water_timeseries.downloader import EarthEngineDownloader
import nest_asyncio
import sys
from pathlib import Path
import asyncio
import datetime
import diagnose_netcdf_2
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


def generate_monthly_first_days(max_current_datetime):
    # Get first day of next month after max_current_datetime
    if max_current_datetime.month == 12:
        start_date = datetime.datetime(max_current_datetime.year + 1, 1, 1)
    else:
        if max_current_datetime.day == 1:
            start_date = datetime.datetime(max_current_datetime.year, max_current_datetime.month + 1, 1)
        else:
            start_date = datetime.datetime(max_current_datetime.year, max_current_datetime.month + 1, 1)

    # Get first day of current month
    today = datetime.datetime.now()
    current_month_first = datetime.datetime(today.year, today.month, 1)

    # Generate list
    result = []
    current = start_date

    while current <= current_month_first:
        result.append(current)
        # Move to next month
        if current.month == 12:
            current = datetime.datetime(current.year + 1, 1, 1)
        else:
            current = datetime.datetime(current.year, current.month + 1, 1)

    return result

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
        most_recent_file_datetimes = diagnose_netcdf_2.get_netcdf_datetimes(most_recent)
        most_recent_datetime = max(most_recent_file_datetimes)
        most_recent_year = most_recent_datetime.year
        most_recent_month = most_recent_datetime.month
        new_datetimes = generate_monthly_first_days(most_recent_datetime)
        new_years = []
        for new_datetime in new_datetimes:
            current_year = new_datetime.year
            new_years.append(current_year)
        # more than 1 year
        new_years.sort()
        new_download_dw_files = []
        if len(new_years) > 2:
            first_year = new_years[0]
            first_year_months = []
            # get months for the first year
            last_year = new_years[1]
            last_year_months = []
            # get months for the last year
            for new_datetime in new_datetimes:
                current_year = new_datetime.year
                if current_year == first_year:
                    first_year_months.append(new_datetime.month)
                if current_year == last_year:
                    last_year_months.append(new_datetime.month)
            print("We have more than 1 new year")
            # TODO get months for first and last year
            # download the first year
            for i in range(0, len(new_years)):
                if i == 0:
                    print("Doing the first year")
                    num_file = str(i + 1)
                    current_download_filename = 'dynamic_world_data_' + CURRENT_TIMESTAMP + '_' + num_file + '_.' + FORMAT
                    dl = EarthEngineDownloader(ee_auth=True, logger=logger)
                    ds = dl.download_dw_monthly(
                        vector_dataset=VECTOR_DATASET,
                        name_attribute="id_geohash",
                        years=[first_year],
                        months=first_year_months,
                        save_to_file=str(current_download_filename),
                    )
                    new_file_exists = os.path.exists(current_download_filename)
                    if new_file_exists:
                        print(f"New file {current_download_filename} exists")
                        readable_filesize = human_file_size(Path(current_download_filename))
                        print(f"New file {current_download_filename} size: {readable_filesize}")
                        new_download_dw_files.append(current_download_filename)
                elif i == len(new_years) - 1:
                    print("Doing the last year")
                    num_file = str(i + 1)
                    current_download_filename = 'dynamic_world_data_' + CURRENT_TIMESTAMP + '_' + num_file + '_.' + FORMAT
                    dl = EarthEngineDownloader(ee_auth=True, logger=logger)
                    ds = dl.download_dw_monthly(
                        vector_dataset=VECTOR_DATASET,
                        name_attribute="id_geohash",
                        years=[first_year],
                        months=first_year_months,
                        save_to_file=str(current_download_filename),
                    )
                    new_file_exists = os.path.exists(current_download_filename)
                    if new_file_exists:
                        print(f"New file {current_download_filename} exists")
                        readable_filesize = human_file_size(Path(current_download_filename))
                        print(f"New file {current_download_filename} size: {readable_filesize}")
                        new_download_dw_files.append(current_download_filename)
                else:
                    print("Doing other years")
            # is the next year the current year? if so, download until the current month
            # is the next year in the future? then download all years in between, 12 months, then the present year, start to now
        elif len(new_years) == 2:
            first_year = new_years[0]
            first_year_months = []
            # get months for the first year
            last_year = new_years[1]
            last_year_months = []
            # get months for the last year
            for new_datetime in new_datetimes:
                current_year = new_datetime.year
                if current_year == first_year:
                    first_year_months.append(new_datetime.month)
                if current_year == last_year:
                    last_year_months.append(new_datetime.month)
            # download the first year
            first_download_filename ='dynamic_world_data_' +CURRENT_TIMESTAMP + '_1_.' + FORMAT
            dl = EarthEngineDownloader(ee_auth=True, logger=logger)
            ds = dl.download_dw_monthly(
                vector_dataset=VECTOR_DATASET,
                name_attribute="id_geohash",
                years=[first_year],
                months=first_year_months,
                save_to_file=str(first_download_filename),
            )
            new_file_exists = os.path.exists(first_download_filename)
            if new_file_exists:
                print(f"New file {first_download_filename} exists")
                readable_filesize = human_file_size(Path(first_download_filename))
                print(f"New file {first_download_filename} size: {readable_filesize}")
                new_download_dw_files.append(first_download_filename)
            # download the second year
            second_download_filename = 'dynamic_world_data_' + CURRENT_TIMESTAMP + '_2_.' + FORMAT
            dl = EarthEngineDownloader(ee_auth=True, logger=logger)
            ds = dl.download_dw_monthly(
                vector_dataset=VECTOR_DATASET,
                name_attribute="id_geohash",
                years=[last_year],
                months=last_year_months,
                save_to_file=str(second_download_filename),
            )
            new_file_exists = os.path.exists(second_download_filename)
            if new_file_exists:
                print(f"New file {second_download_filename} exists")
                readable_filesize = human_file_size(Path(second_download_filename))
                print(f"New file {second_download_filename} size: {readable_filesize}")
                new_download_dw_files.append(second_download_filename)
                 # TODO combine all the files into one file
            else:
                print("We just have 1 near year")
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