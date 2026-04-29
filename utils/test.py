import os
from loguru import logger
from water_timeseries.downloader import EarthEngineDownloader

# Set your EE project (or pass directly as ee_project parameter)
os.environ["EE_PROJECT"] = "pdg-project-406720"

# Create downloader instance
dl = EarthEngineDownloader(ee_auth=True, logger=logger)


ds = dl.download_dw_monthly(
    vector_dataset="{path_to_vector_dataset_file",
    name_attribute="id_geohash",
    years=[2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    months=[6, 7, 8, 9, 10],
    save_to_file="data.zarr",  # Saves to downloads/data.zarr (relative path)
)

print('done')