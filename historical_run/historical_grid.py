from .run_lake_analysis_by_region_date import run_water_timeseries_analysis
import sys
from loguru import logger
from datetime import date


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
    REGION = sys.argv[1]
    logger.debug(f"Beginning historical run")

    logger.debug(f"Historical dates are")
    historical_dates = get_dates(2016, 2025)
    for date in historical_dates:
        logger.debug(f"Doing historical run for {REGION} and date {date}")
        run_water_timeseries_analysis(REGION, date)