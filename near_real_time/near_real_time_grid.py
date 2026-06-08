import geopandas as gpd
import xarray as xr
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import sys
import geemap
import ee
import glob
import os
from water_timeseries.downloader import EarthEngineDownloader
from water_timeseries.utils.spatial import create_longitude_latitude_grid, filter_gdf_by_bbox
from water_timeseries.dataset import DWDataset
from water_timeseries.breakpoint import NRTBreakpoint
import datetime
from region_boundaries import get_region_boundaries
import download_new_dynamic_world_data
import shutil
import json


def main():
    src = '/root/.config/earthengine/credentials'
    logger.debug(f"{src} exists {os.path.exists(src)}")
    dst = '/tmp/ee_creds.json'
    shutil.copy2(src, dst)

    # Read credentials
    with open(dst, 'r') as f:
        credentials = json.load(f)

    # Initialize EE with credentials directly
    ee.Initialize(credentials=credentials, project='pdg-project-406720')
    logger.debug("✓ Earth Engine initialized successfully")

    # Also set up geemap to use the same
    os.environ['EARTHENGINE_TOKEN'] = dst
    os.environ['EE_CONFIG_DIR'] = '/tmp'

    # Create a flag file to indicate initialization is done
    with open('/tmp/ee_initialized', 'w') as f:
        f.write('done')

    region_boundaries = get_region_boundaries()

    start = datetime.datetime.now()
    logger.debug(f"Current time: {datetime.datetime.now()}")

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

    logger.debug(f"Version of geemap is {geemap.__version__}")

    # import geemap
    # import ee
    #
    # # Try different initialization methods based on geemap version
    # if hasattr(geemap, 'ee_initialize'):
    #     geemap.ee_initialize(project=EE_PROJECT_ID)
    # elif hasattr(geemap, 'initialize'):
    #     geemap.initialize(project=EE_PROJECT_ID)
    # else:
    #     ee.Initialize(project=EE_PROJECT_ID)

    current_region = os.getenv('CURRENT_REGION', 'TEST')

    dynamic_world_data_dir = os.environ['dynamic_world_data']
    dynamic_world_download_dir = Path(os.environ['dynamic_world_downloads'])
    dynamic_world_download_dir.mkdir(exist_ok=True, parents=True)
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))

    if not all_dynamic_world_files:
        logger.error(f"No .nc files found in {dynamic_world_data_dir}")
        sys.exit(1)

    # TODO use region here
    bounding_box_coords = region_boundaries['TEST']

    X_MIN_START = bounding_box_coords['X_MIN_START']
    X_MIN_END = bounding_box_coords['X_MIN_END']
    Y_MIN_START = bounding_box_coords['Y_MIN_START']
    Y_MIN_END = bounding_box_coords['Y_MIN_END']

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
    path_lake_vector = vector_lake_file

    ANALYSIS_DATE = "2026-05"

    # read lake vectors
    gdf = gpd.read_parquet(path_lake_vector)

    # read historical DW data
    ds_raw = xr.open_dataset(path_historical_dw)

    bbox_size_lon = 1
    bbox_size_lat = 1
    grid = create_longitude_latitude_grid(lon_range=(X_MIN_START, X_MIN_END), lat_range=(Y_MIN_START, Y_MIN_END),
                                          bbox_size_lon=bbox_size_lon, bbox_size_lat=bbox_size_lat)
    print('created grid')

    bp = NRTBreakpoint()

    # create directory for current data run
    current_breakpoint_dir = Path(output_dir) / f'breakpoint_{ANALYSIS_DATE}'
    current_breakpoint_dir.mkdir(exist_ok=True, parents=True)

    # create directory for partial dynamic world downloads
    current_download_dir = Path(str(dynamic_world_download_dir), f'download_{ANALYSIS_DATE}')
    current_download_dir.mkdir(exist_ok=True, parents=True)

    if not hasattr(geemap, 'ee_initialize'):
        logger.warning("geemap.ee_initialize missing, adding runtime patch")

        def ee_initialize(project=None, **kwargs):
            if project:
                ee.Initialize(project=project, **kwargs)
            else:
                ee.Initialize(**kwargs)

        geemap.ee_initialize = ee_initialize
        logger.info("Runtime patch applied to geemap")

    # setup downloader
    downloader = EarthEngineDownloader(ee_project=EE_PROJECT_ID)

    breaks_list = []
    total = len(grid[:])

    # run loop
    for i, (lon, lat) in enumerate(tqdm(grid[:], total=total, desc="Processing")):
        # setup box
        logger.debug(f"Processing {i}/{total} grid tiles.")
        bbox_west = int(lon)
        bbox_east = int(lon + bbox_size_lon)
        bbox_south = int(lat)
        bbox_north = int(lat + bbox_size_lat)

        print(f"Run processing for bbox: {bbox_west} {bbox_east} {bbox_south} {bbox_north}")

        # setup outfile_download and check if already processed
        outfile_download = current_download_dir / f'DW_{ANALYSIS_DATE}_{bbox_west}_{bbox_east}_{bbox_south}_{bbox_north}.nc'
        outfile_breaks = current_breakpoint_dir / f'DW_{ANALYSIS_DATE}_{bbox_west}_{bbox_east}_{bbox_south}_{bbox_north}_breaks.parquet'

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
        path_to_joined_file = current_breakpoint_dir / f'drain_{ANALYSIS_DATE}.parquet'
        joined.to_parquet(path_to_joined_file)
        breaks_merged.sort_values('drainage_confidence', ascending=False)

    end = datetime.datetime.now()
    logger.debug(f"Finished processing {ANALYSIS_DATE} at time {end}")
    total_time = end - start
    logger.debug(f"Finished in {total_time}")

    logger.info(f"Combining historical and new DW data into a single netcdf file for {ANALYSIS_DATE}")

    # Memory-efficient approach: Process files one by one and write incrementally
    # First, get all downloaded files
    downloaded_files = sorted(glob.glob(str(current_download_dir / f'DW_{ANALYSIS_DATE}_*.nc')))
    output_netcdf = Path(output_dir) / f'lakes_dw_Vd2_{ANALYSIS_DATE}.nc'

    if downloaded_files:
        # Open historical file once to get structure and initial data
        ds_historical = xr.open_dataset(most_recent_dynamic_world_file)

        # Create a list of all datasets to merge (historical + all downloads)
        # Using xr.open_mfdataset with concat_dim to handle combining efficiently
        # But since we need to be careful with memory, we'll process in chunks

        # Open all downloaded files as a single dataset using multi-file dataset
        # This doesn't load data into memory until needed
        ds_downloads = xr.open_mfdataset(downloaded_files, combine='by_coords')

        # Merge historical with downloads
        # xr.merge creates a new dataset but doesn't load data into memory
        ds_combined = xr.merge([ds_historical, ds_downloads], join='outer')

        # Write to netcdf with compression to save space
        # Use encoding to compress data and reduce file size
        encoding = {var: {'zlib': True, 'complevel': 5} for var in ds_combined.data_vars}

        # Write to temporary file first to avoid partial writes
        temp_output = output_netcdf.with_suffix('.tmp.nc')
        ds_combined.to_netcdf(temp_output, encoding=encoding, mode='w')

        # Close all open datasets to free memory
        ds_historical.close()
        ds_downloads.close()
        ds_combined.close()

        # Replace with final file
        temp_output.rename(output_netcdf)

        logger.info(f"Successfully created combined netcdf: {output_netcdf}")
        logger.info(f"File size: {output_netcdf.stat().st_size / (1024 ** 3):.2f} GB")
    else:
        logger.warning(f"No downloaded files found in {current_download_dir}")

    # Clean up memory
    del ds_historical, ds_downloads, ds_combined

    logger.info("Combining breakpoint parquet files")

    # Get all breakpoint parquet files
    break_files = sorted(glob.glob(str(current_breakpoint_dir / f'DW_{ANALYSIS_DATE}_*_breaks.parquet')))

    if break_files:
        # Memory-efficient approach: Read and combine parquet files in chunks
        # For parquet, we can use pandas with chunking or dask for very large datasets

        # Option 1: If the combined file might be large but fits in memory
        # Read all files and concatenate
        dfs = []
        total_rows = 0

        for file in tqdm(break_files, desc="Reading breakpoint files"):
            df = pd.read_parquet(file)
            dfs.append(df)
            total_rows += len(df)

            # Optional: Log progress
            logger.debug(f"Read {file} with {len(df)} rows. Total rows so far: {total_rows}")

        # Concatenate all dataframes
        if dfs:
            breaks_combined = pd.concat(dfs, ignore_index=True)

            # Optional: Sort by drainage confidence as in your original code
            breaks_combined = breaks_combined.sort_values('drainage_confidence', ascending=False)

            # Create output filename with bounding box coordinates
            output_parquet = Path(
                output_dir) / f'DW_{ANALYSIS_DATE}_{X_MIN_START}_{X_MIN_END}_{Y_MIN_START}_{Y_MIN_END}.parquet'

            # Save combined parquet file
            breaks_combined.to_parquet(output_parquet, index=False)

            logger.info(f"Successfully created combined parquet: {output_parquet}")
            logger.info(f"Total rows: {len(breaks_combined)}")
            logger.info(f"File size: {output_parquet.stat().st_size / (1024 ** 2):.2f} MB")

            # Clean up
            del dfs, breaks_combined
        else:
            logger.warning("No data found in breakpoint files")
    else:
        logger.warning(f"No breakpoint files found in {current_breakpoint_dir}")


if __name__ == '__main__':
    main()


