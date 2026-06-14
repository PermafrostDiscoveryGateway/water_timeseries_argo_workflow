from run_lake_analysis_by_region_date import run_water_timeseries_analysis
import sys
from loguru import logger
from datetime import date
from dotenv import load_dotenv
import os
import utils.region_boundaries

def get_dates(start_year, end_year):
    """
    Generate dates for months 6,7,8,9 (June-September) between start_year and end_year.

    Args:
        start_year: Starting year (inclusive)
        end_year: Ending year (inclusive)

    Returns:
        List of date strings in format 'YYYY-MM-DD' (always the 1st of each month)
    """
    dates = []

    for year in range(start_year, end_year + 1):
        for month in [6, 7, 8, 9]:
            # Use day=1 for the first of each month
            date_obj = date(year, month, 1)
            dates.append(date_obj.strftime('%Y-%m-%d'))

    return dates


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
    start_year = int(os.getenv("START_YEAR", 2016))
    end_year = int(os.getenv("END_YEAR", 2025))


    logger.debug(f"Historical dates are")
    historical_dates = get_dates(start_year, end_year)
    for REGION in REGION_NAMES:
        for date in historical_dates:
            logger.debug(f"Doing historical run for {REGION} and date {date}")
            run_water_timeseries_analysis(REGION, date)

if __name__ == "__main__":
    main()