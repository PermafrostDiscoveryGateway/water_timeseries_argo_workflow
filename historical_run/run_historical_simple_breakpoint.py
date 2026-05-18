#!/usr/bin/env python3
"""
Run historical Simple breakpoint analysis on lake water time series data.

This script uses a rolling window statistical approach to detect significant
drops in lake water area over a historical time period.

Usage:
    python run_historical_simple_breakpoint.py [env_file_path]

Example:
    python run_historical_simple_breakpoint.py .env
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

# Historical analysis date range
START_DATE = "2020-01-01"  # Format: YYYY-MM-DD
END_DATE = "2023-12-31"  # Format: YYYY-MM-DD

# Simple breakpoint parameters
THRESHOLD = -0.25  # Threshold for detecting a break (negative values indicate drop)
WINDOW = 3  # Rolling window size (number of observations)
METHOD = "median"  # Rolling statistic: "mean", "median", or "max"

# Processing parameters
LAKE_CHUNK_SIZE = 2000  # Number of lakes to process in each chunk (Simple is faster)
SAVE_INTERMEDIATE = True  # Save intermediate chunk results

# Optional: Filter to specific lakes (set to None for all lakes)
# Example: SPECIFIC_LAKES = ["geohash1", "geohash2", "geohash3"]
SPECIFIC_LAKES = None

# Output directory (relative or absolute path)
OUTPUT_DIR = "./historical_simple_results"


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

def run_historical_simple_breakpoint(
        input_nc_file: Path,
        start_date: str,
        end_date: str,
        output_dir: Path,
        threshold: float = -0.25,
        window: int = 3,
        method: str = "median",
        lake_chunk_size: int = 2000,
        specific_lakes: list = None,
        save_intermediate: bool = True
) -> pd.DataFrame:
    """
    Run historical Simple breakpoint analysis on lake data.

    Parameters
    ----------
    input_nc_file : Path
        Path to the NetCDF file containing lake data
    start_date : str
        Start date for analysis (YYYY-MM-DD)
    end_date : str
        End date for analysis (YYYY-MM-DD)
    output_dir : Path
        Directory to save results
    threshold : float
        Threshold for detecting a break (negative values indicate water loss)
    window : int
        Rolling window size for calculating statistics
    method : str
        Rolling statistic to use: "mean", "median", or "max"
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

    # Validate method
    if method not in ["mean", "median", "max"]:
        raise ValueError(f"Method must be 'mean', 'median', or 'max', got '{method}'")

    # Log configuration
    logger.info("=" * 80)
    logger.info("HISTORICAL SIMPLE BREAKPOINT ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Input file: {input_nc_file}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Window size: {window}")
    logger.info(f"Method: {method}")
    logger.info(f"Lake chunk size: {lake_chunk_size}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    # Load dataset
    logger.info("Loading dataset...")
    ds = xr.open_dataset(input_nc_file)
    dw_dataset = DWDataset(ds)

    logger.info(f"Dataset contains {len(dw_dataset.object_ids_)} lakes")
    logger.info(f"Date range in dataset: {dw_dataset.dates_[0]} to {dw_dataset.dates_[-1]}")

    # Initialize Simple analyzer
    analyzer = HistoricalBreakpointAnalyzer(
        method="simple",
        threshold=threshold,
        window=window,
        method_name=method  # Pass the rolling statistic method
    )

    # Run analysis
    logger.info("Starting Simple breakpoint detection...")
    results = analyzer.analyze_time_range(
        dataset=dw_dataset,
        start_date=start_date,
        end_date=end_date,
        object_ids=specific_lakes,
        lake_chunk_size=lake_chunk_size,
        save_intermediate=save_intermediate,
        output_dir=output_dir
    )

    # Save and summarize results
    if not results.empty:
        # Save main results files
        csv_file = output_dir / f"simple_breakpoints_{start_date}_to_{end_date}.csv"
        parquet_file = output_dir / f"simple_breakpoints_{start_date}_to_{end_date}.parquet"

        results.to_csv(csv_file)
        results.to_parquet(parquet_file)

        # Calculate additional statistics
        stats = {
            'total_breakpoints': len(results),
            'unique_lakes': results.index.nunique(),
            'date_range': {
                'earliest_break': results['date_break'].min() if 'date_break' in results.columns else None,
                'latest_break': results['date_break'].max() if 'date_break' in results.columns else None
            }
        }

        # Calculate water area changes if columns exist
        if 'water_area_before' in results.columns and 'water_area_after' in results.columns:
            results['water_area_change'] = results['water_area_after'] - results['water_area_before']
            results['water_area_change_pct'] = (results['water_area_change'] / results['water_area_before']) * 100
            stats['mean_water_area_change'] = float(results['water_area_change'].mean())
            stats['mean_water_area_change_pct'] = float(results['water_area_change_pct'].mean())

        # Save summary
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'method': 'simple',
            'start_date': start_date,
            'end_date': end_date,
            'threshold': threshold,
            'window': window,
            'method_type': method,
            **stats,
            'output_files': {
                'csv': str(csv_file),
                'parquet': str(parquet_file)
            }
        }

        import json
        summary_file = output_dir / f"simple_analysis_summary_{start_date}_to_{end_date}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS COMPLETE - SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total breakpoints found: {len(results)}")
        logger.info(f"Unique lakes with breaks: {results.index.nunique()}")

        if 'water_area_change' in results.columns:
            logger.info(f"Mean water area change: {results['water_area_change'].mean():.2f} km²")
            logger.info(f"Mean change percentage: {results['water_area_change_pct'].mean():.1f}%")

        logger.info(f"\nResults saved to:")
        logger.info(f"  CSV: {csv_file}")
        logger.info(f"  Parquet: {parquet_file}")
        logger.info(f"  Summary: {summary_file}")
        logger.info("=" * 80)

        return results
    else:
        logger.warning("No breakpoints found in the specified time range")
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
        results = run_historical_simple_breakpoint(
            input_nc_file=input_file,
            start_date=START_DATE,
            end_date=END_DATE,
            output_dir=Path(OUTPUT_DIR),
            threshold=THRESHOLD,
            window=WINDOW,
            method=METHOD,
            lake_chunk_size=LAKE_CHUNK_SIZE,
            specific_lakes=SPECIFIC_LAKES,
            save_intermediate=SAVE_INTERMEDIATE
        )

        logger.info("✅ Simple breakpoint analysis completed successfully!")

        # Optional: Print first few results as preview
        if not results.empty:
            logger.info("\n📊 Preview of results (first 5 rows):")
            logger.info(results.head().to_string())

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()