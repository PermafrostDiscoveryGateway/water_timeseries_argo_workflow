import os
from loguru import logger
from water_timeseries.downloader import EarthEngineDownloader

# Set your EE project (or pass directly as ee_project parameter)
os.environ["EE_PROJECT"] = "pdg-project-406720"

# Create downloader instance
dl = EarthEngineDownloader(ee_auth=True, logger=logger)


ds = dl.download_dw_monthly(
    vector_dataset="/Users/helium/ncsa/pdg/water-timeseries-v2/tests/data/Nitze_etal_Lakes_filtered_full_set_V2d.parquet",
    name_attribute="id_geohash",
    years=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    months=[6, 7, 8, 9, 10],
    save_to_file="data_download.zarr",  # Saves to downloads/data.zarr (relative path)
)

print('done')

