#!/usr/bin/env python3
"""
Run historical BEAST breakpoint analysis on lake water time series data.

This script performs Bayesian changepoint detection using the RBEAST library
to identify significant changes in lake water area at specific analysis dates.

Usage:
    python run_historical_beast_breakpoint.py [env_file_path]

Example:
    python run_historical_beast_breakpoint.py .env
"""

import os
import sys
import glob
from pathlib import Path
from datetime import datetime
from loguru import logger
import xarray as xr
import pandas as pd

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_timeseries.dataset import DWDataset
from historical_breakpoint_analyzer import HistoricalBreakpointAnalyzer

# ============================================================================
# CONFIGURATION - Edit these values as needed
# ============================================================================

# Analysis dates - List of specific dates to analyze
# Format: YYYY-MM-DD
ANALYSIS_DATES = [
    "2020-06-01",
    "2020-12-01",
    "2021-06-01",
    "2021-12-01",
    "2022-06-01",
    "2022-12-01",
    "2023-06-01",
    "2023-12-01",
]

# Alternative: Use date range with step months
# If USE_DATE_RANGE = True, the above ANALYSIS_DATES will be ignored
USE_DATE_RANGE = False
START_DATE = "2020-01-01"  # Format: YYYY-MM-DD
END_DATE = "2023-12-31"  # Format: YYYY-MM-DD
STEP_MONTHS = 3  # Number of months between analyses (3 = quarterly)

# BEAST-specific parameters
BREAK_THRESHOLD = 0.5  # Probability threshold for detecting a break (0-1)
TREND_MAX_ORDER = 0  # Maximum order of trend component (0 = no trend)
TREND_MIN_SEP_DIST = 1  # Minimum separation between change points

# Processing parameters
LAKE_CHUNK_SIZE = 500  # Number of lakes to process in each chunk
SAVE_INTERMEDIATE = True  # Save intermediate chunk results

# Optional: Filter to specific lakes (set to None for all lakes)
# Example: SPECIFIC_LAKES = ["geohash1", "geohash2", "geohash3"]
SPECIFIC_LAKES = None

# Output directory (relative or absolute path)
OUTPUT_DIR = "./historical_beast_results"


# ============================================================================
# ENVIRONMENT LOADING
# ============================================================================

def load_environment(env_path=None):
    """
    Load environment variables from .env file or use existing.

    Parameters
    ----------
    env_path : str, optional
        Path to .env file (can be provided as command line argument)
    """
    from dotenv import load_dotenv

    if env_path and Path(env_path).exists():
        load_dotenv(dotenv_path=env_path, override=False)
        logger.info(f"Loaded environment from: {env_path}")
    else:
        # Try default .env file
        default_env = Path.cwd() / ".env"
        if default_env.exists():
            load_dotenv(dotenv_path=default_env, override=False)
            logger.info(f"Loaded environment from default .env: {default_env}")
        else:
            logger.info("Using existing environment variables")

    # Validate required environment variables
    required_vars = ['dynamic_world_dir']
    missing_vars = [var for var in required_vars if var not in os.environ]

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        raise EnvironmentError(f"Missing required environment variables: {missing_vars}")

    # Optional: Set Google Earth Engine project if provided
    if 'project' in os.environ:
        os.environ["EE_PROJECT"] = os.environ['project']
        logger.info(f"Set EE_PROJECT to {os.environ['project']}")


def get_input_file():
    """
    Get the input NetCDF file from environment or find most recent.

    Returns
    -------
    Path
        Path to the input NetCDF file
    """
    dynamic_world_dir = Path(os.environ['dynamic_world_dir'])

    # Check if specific file is specified in environment
    if 'dynamic_world_data_file' in os.environ:
        input_file = Path(os.environ['dynamic_world_data_file'])
        if input_file.exists():
            logger.info(f"Using specified file: {input_file}")
            return input_file
        else:
            logger.warning(f"Specified file not found: {input_file}")

    # Find all NetCDF files in the directory
    nc_files = list(dynamic_world_dir.glob("*.nc"))

    if not nc_files:
        logger.error(f"No .nc files found in {dynamic_world_dir}")
        raise FileNotFoundError(f"No .nc files found in {dynamic_world_dir}")

    # Use the most recent file
    input_file = max(nc_files, key=lambda f: f.stat().st_ctime)
    logger.info(f"Using most recent file: {input_file}")

    return input_file


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_historical_beast_breakpoint(
        input_nc_file: Path,
        output_dir: Path,
        analysis_dates: List[str] = None,
        use_date_range: bool = False,
        start_date: str = None,
        end_date: str = None,
        step_months: int = 3,
        break_threshold: float = 0.5,
        trend_max_order: int = 0,
        trend_min_sep_dist: int = 1,
        lake_chunk_size: int = 500,
        specific_lakes: list = None,
        save_intermediate: bool = True
) -> pd.DataFrame:
    """
    Run historical BEAST breakpoint analysis on lake data.

    Parameters
    ----------
    input_nc_file : Path
        Path to the NetCDF file containing lake data
    output_dir : Path
        Directory to save results
    analysis_dates : List[str], optional
        List of specific dates to analyze (YYYY-MM-DD)
    use_date_range : bool
        If True, generate dates from start_date to end_date with step_months
    start_date : str, optional
        Start date for date range (required if use_date_range=True)
    end_date : str, optional
        End date for date range (required if use_date_range=True)
    step_months : int
        Number of months between analyses (default: 3)
    break_threshold : float
        Probability threshold for break detection (0-1)
    trend_max_order : int
        Maximum order of trend component for BEAST
    trend_min_sep_dist : int
        Minimum separation between change points
    lake_chunk_size : int
        Number of lakes to process in each chunk
    specific_lakes : list, optional
        List of specific lake IDs to analyze (None for all)
    save_intermediate : bool
        Whether to save intermediate chunk results

    Returns
    -------
    pd.DataFrame
        DataFrame containing breakpoint results
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine analysis dates
    if use_date_range:
        if not start_date or not end_date:
            raise ValueError("start_date and end_date are required when use_date_range=True")
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        analysis_dates = pd.date_range(start=start, end=end, freq=f'{step_months}MS')
        logger.info(f"Generated {len(analysis_dates)} analysis dates from {start_date} to {end_date}")
    else:
        if not analysis_dates:
            raise ValueError("Either analysis_dates or use_date_range must be provided")
        analysis_dates = [pd.to_datetime(date) for date in analysis_dates]
        logger.info(f"Using {len(analysis_dates)} specified analysis dates")

    # Log configuration
    logger.info("=" * 80)
    logger.info("HISTORICAL BEAST BREAKPOINT ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Input file: {input_nc_file}")
    logger.info(f"Analysis dates: {[d.strftime('%Y-%m-%d') for d in analysis_dates]}")
    logger.info(f"Break threshold: {break_threshold}")
    logger.info(f"Trend max order: {trend_max_order}")
    logger.info(f"Trend min separation: {trend_min_sep_dist}")
    logger.info(f"Lake chunk size: {lake_chunk_size}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    # Load dataset
    logger.info("Loading dataset...")
    ds = xr.open_dataset(input_nc_file)
    dw_dataset = DWDataset(ds)

    logger.info(f"Dataset contains {len(dw_dataset.object_ids_)} lakes")
    logger.info(f"Date range in dataset: {dw_dataset.dates_[0]} to {dw_dataset.dates_[-1]}")

    # Initialize BEAST analyzer
    analyzer = HistoricalBreakpointAnalyzer(
        method="beast",
        break_threshold=break_threshold,
        trendMaxOrder=trend_max_order,
        trendMinSepDist=trend_min_sep_dist
    )

    # Run analysis
    logger.info("Starting BEAST breakpoint detection...")
    results = analyzer.analyze_dates(
        dataset=dw_dataset,
        analysis_dates=analysis_dates,
        object_ids=specific_lakes,
        lake_chunk_size=lake_chunk_size,
        save_intermediate=save_intermediate,
        output_dir=output_dir
    )

    # Save and summarize results
    if not results.empty:
        # Create filename based on date range or specific dates
        if use_date_range:
            date_str = f"{start_date}_to_{end_date}_step{step_months}m"
        else:
            date_str = f"{analysis_dates[0].strftime('%Y%m%d')}_to_{analysis_dates[-1].strftime('%Y%m%d')}_{len(analysis_dates)}dates"

        # Save main results files
        csv_file = output_dir / f"beast_breakpoints_{date_str}.csv"
        parquet_file = output_dir / f"beast_breakpoints_{date_str}.parquet"

        results.to_csv(csv_file)
        results.to_parquet(parquet_file)

        # Save summary statistics
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'method': 'beast',
            'analysis_dates': [d.strftime('%Y-%m-%d') for d in analysis_dates],
            'break_threshold': break_threshold,
            'total_breakpoints_found': len(results),
            'unique_lakes_with_breaks': results.index.nunique(),
            'unique_analysis_dates': results['analysis_date'].nunique() if 'analysis_date' in results.columns else len(
                analysis_dates),
            'output_files': {
                'csv': str(csv_file),
                'parquet': str(parquet_file)
            }
        }

        # Add BEAST-specific statistics
        if 'proba_rbeast' in results.columns:
            summary['mean_break_probability'] = float(results['proba_rbeast'].mean())
            summary['max_break_probability'] = float(results['proba_rbeast'].max())
            summary['min_break_probability'] = float(results['proba_rbeast'].min())

        if 'break_number' in results.columns:
            summary['total_break_events'] = int(results['break_number'].max())

        # Save summary to JSON
        import json
        summary_file = output_dir / f"beast_analysis_summary_{date_str}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS COMPLETE - SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total breakpoints found: {len(results)}")
        logger.info(f"Unique lakes with breaks: {results.index.nunique()}")
        logger.info(
            f"Analysis dates processed: {results['analysis_date'].nunique() if 'analysis_date' in results.columns else len(analysis_dates)}")

        if 'proba_rbeast' in results.columns:
            logger.info(f"Mean break probability: {results['proba_rbeast'].mean():.3f}")
            logger.info(
                f"Break probability range: {results['proba_rbeast'].min():.3f} - {results['proba_rbeast'].max():.3f}")

        logger.info(f"\nResults saved to:")
        logger.info(f"  CSV: {csv_file}")
        logger.info(f"  Parquet: {parquet_file}")
        logger.info(f"  Summary: {summary_file}")
        logger.info("=" * 80)

        return results
    else:
        logger.warning("No breakpoints found for any analysis date")
        return pd.DataFrame()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    env_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Load environment
    try:
        load_environment(env_path)
    except Exception as e:
        logger.error(f"Failed to load environment: {e}")
        sys.exit(1)

    # Get input file
    try:
        input_file = get_input_file()
    except Exception as e:
        logger.error(f"Failed to get input file: {e}")
        sys.exit(1)

    # Run analysis
    try:
        results = run_historical_beast_breakpoint(
            input_nc_file=input_file,
            output_dir=Path(OUTPUT_DIR),
            analysis_dates=ANALYSIS_DATES,
            use_date_range=USE_DATE_RANGE,
            start_date=START_DATE,
            end_date=END_DATE,
            step_months=STEP_MONTHS,
            break_threshold=BREAK_THRESHOLD,
            trend_max_order=TREND_MAX_ORDER,
            trend_min_sep_dist=TREND_MIN_SEP_DIST,
            lake_chunk_size=LAKE_CHUNK_SIZE,
            specific_lakes=SPECIFIC_LAKES,
            save_intermediate=SAVE_INTERMEDIATE
        )

        logger.info("✅ BEAST breakpoint analysis completed successfully!")

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()