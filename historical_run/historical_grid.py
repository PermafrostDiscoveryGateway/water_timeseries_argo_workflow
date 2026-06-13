import run_lake_analysis_by_region_date
import sys
from loguru import logger
from datetime import date
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
            dates.append(date_obj.strftime('%Y-%m'))

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

    REGION = os.getenv("REGION", "EURASIA3")
    start_year = int(os.getenv("START_YEAR", 2017))
    end_year = int(os.getenv("END_YEAR", 2020))


    logger.debug(f"Historical dates are")
    historical_dates = get_dates(start_year, end_year)
    for date in historical_dates:
        logger.debug(f"Doing historical run for {REGION} and date {date}")
        run_lake_analysis_by_region_date.run_water_timeseries_analysis(REGION, date)

if __name__ == "__main__":
    main()