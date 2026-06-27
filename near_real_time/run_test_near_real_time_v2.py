from near_real_time_grid import near_real_time_region
import sys
from loguru import logger
import glob
from datetime import date
from dotenv import load_dotenv
import os
import utils.region_boundaries

def main():
    logger.debug(f"Beginning historical run")
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    dynamic_world_data_dir = os.environ['dynamic_world_data']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))


    near_real_time_region(region='TEST')


if __name__ == "__main__":
    main()