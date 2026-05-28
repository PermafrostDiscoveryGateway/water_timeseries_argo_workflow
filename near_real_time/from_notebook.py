import geopandas as gpd
import xarray as xr
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import sys
import glob
import os
import download_new_dynamic_world_data
from water_timeseries.downloader import EarthEngineDownloader
from water_timeseries.utils.spatial import create_longitude_latitude_grid, filter_gdf_by_bbox
from water_timeseries.dataset import DWDataset
from water_timeseries.breakpoint import NRTBreakpoint

if len(sys.argv) > 1:
    env_path = sys.argv[1]
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loading environment from: {env_path}")
else:
    load_dotenv()
    logger.info("Loading environment from default .env file")

output_dir = os.environ['output_dir']
project = os.environ['project']
EE_PROJECT_ID = project
os.environ["EE_PROJECT"] = EE_PROJECT_ID

dynamic_world_dir = os.environ['dynamic_world_data']
all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
if not all_dynamic_world_files:
    logger.error(f"No .nc files found in {dynamic_world_dir}")
    sys.exit(1)

most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)

missing_dates = download_new_dynamic_world_data.check_missing_data_in_netcdf(most_recent_dynamic_world_file, )
missing_analysis_dates = []
for date in missing_dates:
    missing_date_string = date.strftime("%Y-%m")
    logger.debug(f"We are missing {missing_date_string}")
    missing_analysis_dates.append(missing_date_string)
vector_lake_file = os.environ['vector_lake_file']

# lake vector path
path_historical_dw = most_recent_dynamic_world_file
# historical DW data path
path_lake_vector= vector_lake_file

ANALYSIS_DATE = "2026-05"

# read lake vectors
gdf = gpd.read_parquet(path_lake_vector)

# read historical DW data
ds_raw = xr.open_dataset(path_historical_dw)

bbox_size_lon = 1
bbox_size_lat = 1
grid = create_longitude_latitude_grid(lon_range=(-158, -155), lat_range=(63,66), bbox_size_lon=bbox_size_lon, bbox_size_lat=bbox_size_lat)
print('created grid')

bp = NRTBreakpoint()

outdir = Path(f'download_{ANALYSIS_DATE}')
outdir.mkdir(exist_ok=True, parents=True)
# setup downloader
downloader = EarthEngineDownloader(ee_project='YOUR_EE_PROJECT')

breaks_list = []

# run loop
for lon, lat in tqdm(grid[:]):
    # setup box
    bbox_west = int(lon)
    bbox_east = int(lon + bbox_size_lon)
    bbox_south = int(lat)
    bbox_north = int(lat + bbox_size_lat)

    print(f"Run processing for bbox: {bbox_west} {bbox_east} {bbox_south} {bbox_north}")

    # setup outfile_download and check if already processed
    outfile_download = outdir / f'DW_{ANALYSIS_DATE}_{bbox_west}_{bbox_east}_{bbox_south}_{bbox_north}.nc'
    outfile_breaks = outdir / f'DW_{ANALYSIS_DATE}_{bbox_west}_{bbox_east}_{bbox_south}_{bbox_north}_breaks.parquet'

    # check if breakpoint file already exists
    if outfile_breaks.exists():
        print(f'Breakpoints have been already calculated!: Skip processing for  {bbox_west} {bbox_south} \n')
        print('Data is loaded and appended \n')
        breaks_list.append(pd.read_parquet(outfile_breaks))
        continue

    # subset lakes to grid cell
    gdf_subset = filter_gdf_by_bbox(gdf=gdf, bbox_west=lon, bbox_east=lon + bbox_size_lon, bbox_south=lat,
                                    bbox_north=lat + bbox_size_lat)
    n_lakes = len(gdf_subset)
    print('Number of lakes: ', n_lakes)

    # extract lake ids
    id_list = gdf_subset['id_geohash'].values.tolist()
    if n_lakes == 0:
        print(f'No lakes available for grid {bbox_west} {bbox_south}. Skipping this grid cell! \n')
        continue

    # download
    if not outfile_download.exists():
        # start download for specified date, run up to 2 parallel runs, max_total_requests can be tuned, setting too high might crash the download (rejection by GEE)
        try:
            ds_dl = downloader.download_dw_monthly(gdf=gdf_subset, max_total_requests=2000, n_parallel=2,
                                                   date_list=[ANALYSIS_DATE], save_to_file=outfile_download)
        except ValueError as e:
            expected_msg = "No data was extracted from any chunk. Check GEE request parameters."
            if str(e) == expected_msg:
                print(f"Expected error caught: {e}")
                continue
            else:
                raise
    else:
        print(f'Outfile already exists: Skipping download for {bbox_west} {bbox_south} \n')

    # subset historical data to grid cell
    ds_historical_subset = ds_raw.sel(id_geohash=id_list)

    # merge historical and recent and convert to DWDataset object (required to run breakpoint)
    ds_merged = xr.merge([ds_historical_subset, ds_dl]).sortby('date')
    dwds = DWDataset(ds_merged)

    # run breakpoint analysis
    breaks = bp.calculate_break(dataset=dwds, analysis_date=ANALYSIS_DATE)
    breaks.to_parquet(outfile_breaks)

    # add to merge list
    breaks_list.append(breaks)

    breaks_merged = pd.concat(breaks_list)

    joined = gdf.set_index('id_geohash').join(breaks_merged, how='inner').reset_index()

    joined.to_parquet(f'drain_{ANALYSIS_DATE}.parquet')
    breaks_merged.sort_values('drainage_confidence', ascending=False)

