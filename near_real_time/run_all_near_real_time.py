from near_real_time_grid import near_real_time_region
import sys
from loguru import logger
from datetime import date
from dotenv import load_dotenv
import os
import utils.region_boundaries
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def main():
    logger.debug(f"Beginning historical run")
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    REGIONS = utils.region_boundaries.get_region_boundaries()
    REGION_NAMES = list(REGIONS.keys())

    # TODO get today's date
    # TODO run last month if it is a summer month, and today is after the third

    for REGION in REGION_NAMES:
        # TODO check if this region has been downloaded
        # TODO download here
        near_real_time_region(region=REGION)
    for REGION in REGION_NAMES:
        logger.debug(f"Processing region {REGION}")
        # TODO check if we should process

    logger.debug("Finished NRT for all regions")


if __name__ == "__main__":
    main()