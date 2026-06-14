#!/usr/bin/env python3
import sys
from pathlib import Path
import run_lake_analysis_by_region_date

# def setup_water_timeseries_path():
#     """
#     Dynamically add water-timeseries-v2 to Python path.
#     Works from any location without hardcoded paths.
#     """
#     # Get the current script's location
#     script_path = Path(__file__).resolve()
#
#     # Look for water-timeseries-v2 by traversing up from script location
#     # and checking common sibling/parent locations
#     candidates = []
#
#     # Add current script's directory and parents
#     for parent in [script_path.parent] + list(script_path.parents):
#         candidates.append(parent / "water-timeseries-v2")
#         candidates.append(parent.parent / "water-timeseries-v2")
#         candidates.append(parent / "../water-timeseries-v2")
#
#     # Add common locations relative to home
#     candidates.append(Path.home() / "water-timeseries-v2")
#
#     # Check if the package is already importable
#     try:
#         import water_timeseries
#         print(f"✓ water_timeseries already available at: {water_timeseries.__file__}")
#         return True
#     except ImportError:
#         pass
#
#     # Try each candidate
#     for candidate in candidates:
#         src_path = candidate.resolve() / "src"
#         if src_path.exists() and (src_path / "water_timeseries").exists():
#             print(f"✓ Found water_timeseries at: {src_path}")
#             sys.path.insert(0, str(src_path))
#             return True
#
#     print("⚠️  WARNING: Could not find water-timeseries-v2")
#     print("   Searched in:")
#     for c in candidates[:5]:
#         print(f"     - {c}")
#     return False
#
#
# # Run the setup
# if not setup_water_timeseries_path():
#     print("ERROR: water_timeseries package not found. Please check installation.")
#     sys.exit(1)


import sys
from loguru import logger
from datetime import date
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# Add this at the VERY TOP of your script, before any imports
import sys
import os
from pathlib import Path

# print("=== DIAGNOSTIC INFO ===")
# print(f"Script location: {__file__}")
# print(f"Current working directory: {os.getcwd()}")
# print(f"Python executable: {sys.executable}")
# print(f"Python path: {sys.path[:3]}...")
#
# # Try to find the project root dynamically
# script_path = Path(__file__).resolve()
# print(f"Resolved script path: {script_path}")
#
# # Look for water_timeseries directory
# for parent in [script_path.parent] + list(script_path.parents):
#     print(f"Checking: {parent}")
#     if (parent / "water_timeseries").exists():
#         print(f"✓ Found water_timeseries at: {parent / 'water_timeseries'}")
#         if str(parent) not in sys.path:
#             sys.path.insert(0, str(parent))
#             print(f"Added {parent} to sys.path")
#         break
#     if (parent / "src" / "water_timeseries").exists():
#         print(f"✓ Found water_timeseries at: {parent / 'src' / 'water_timeseries'}")
#         src_path = parent / "src"
#         if str(src_path) not in sys.path:
#             sys.path.insert(0, str(src_path))
#             print(f"Added {src_path} to sys.path")
#         break
#
# print("=== END DIAGNOSTIC ===\n")
# import run_lake_analysis_by_region_date
# # Add project root to Python path
# PROJECT_ROOT = Path("/home/ext_tcnichol_illinois_edu/water-timeseries-v2")
# SRC_PATH = PROJECT_ROOT / "src"
#
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))
# if str(SRC_PATH) not in sys.path:
#     sys.path.insert(0, str(SRC_PATH))


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

    REGION = os.getenv("REGION", "TEST")
    start_year = int(os.getenv("START_YEAR", 2015))
    end_year = int(os.getenv("END_YEAR", 2024))


    logger.debug(f"Historical dates are")
    historical_dates = get_dates(start_year, end_year)
    for date in historical_dates:
        logger.debug(f"Doing historical run for {REGION} and date {date}")
        run_lake_analysis_by_region_date.run_water_timeseries_analysis(REGION, date)

if __name__ == "__main__":
    main()