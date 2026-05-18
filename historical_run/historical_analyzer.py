import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Optional, List, Union
from loguru import logger
from datetime import datetime

from water_timeseries.breakpoint import SimpleBreakpoint, BeastBreakpoint
from water_timeseries.dataset import DWDataset


class HistoricalBreakpointAnalyzer:
    """Base class for historical breakpoint analysis.

    This class provides a unified interface for running Simple and Beast
    breakpoint detection on historical time series data.
    """

    def __init__(
            self,
            method: str = "simple",  # "simple" or "beast"
            threshold: Optional[float] = None,
            window: int = 3,
            break_threshold: float = 0.5,
            **kwargs
    ):
        """Initialize historical breakpoint analyzer.

        Parameters
        ----------
        method : str
            Breakpoint method to use: "simple" or "beast"
        threshold : float, optional
            Threshold for SimpleBreakpoint (default: -0.25)
        window : int
            Window size for SimpleBreakpoint (default: 3)
        break_threshold : float
            Probability threshold for BeastBreakpoint (default: 0.5)
        **kwargs
            Additional arguments passed to the breakpoint method
        """
        self.method = method
        self.threshold = threshold if threshold is not None else -0.25
        self.window = window
        self.break_threshold = break_threshold

        # Initialize the appropriate breakpoint detector
        if method == "simple":
            self.detector = SimpleBreakpoint(
                kwargs_break=dict(
                    window=self.window,
                    method="median",  # Can be parameterized if needed
                    threshold=self.threshold
                )
            )
        elif method == "beast":
            self.detector = BeastBreakpoint(
                kwargs_break=dict(
                    trendMaxOrder=kwargs.get('trendMaxOrder', 0),
                    trendMinSepDist=kwargs.get('trendMinSepDist', 1)
                ),
                break_threshold=self.break_threshold
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'simple' or 'beast'")

    def analyze_time_range(
            self,
            dataset: DWDataset,
            start_date: Union[str, pd.Timestamp, datetime],
            end_date: Union[str, pd.Timestamp, datetime],
            object_ids: Optional[List[str]] = None,
            lake_chunk_size: int = 1000,
            save_intermediate: bool = True,
            output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """Analyze breakpoints over a historical time range.

        Parameters
        ----------
        dataset : DWDataset
            Dataset containing lake water-area data
        start_date : str or pd.Timestamp or datetime
            Start date for historical analysis
        end_date : str or pd.Timestamp or datetime
            End date for historical analysis
        object_ids : List[str], optional
            Specific lake IDs to analyze. If None, analyze all lakes.
        lake_chunk_size : int
            Number of lakes to process in each chunk
        save_intermediate : bool
            Whether to save intermediate results
        output_dir : Path, optional
            Directory to save results (required if save_intermediate=True)

        Returns
        -------
        pd.DataFrame
            DataFrame containing breakpoint information for the time range
        """
        # Convert dates to pandas Timestamp
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        logger.info(f"Analyzing breakpoints from {start_date} to {end_date}")
        logger.info(f"Using method: {self.method}")

        # Filter dataset to time range
        filtered_ds = dataset.ds.sel(date=slice(start_date, end_date))
        filtered_dataset = DWDataset(filtered_ds)

        # Determine which lakes to process
        if object_ids is None:
            object_ids = filtered_dataset.object_ids_
        else:
            # Filter to only include lakes that exist in dataset
            object_ids = [oid for oid in object_ids if oid in filtered_dataset.object_ids_]

        total_lakes = len(object_ids)
        logger.info(f"Processing {total_lakes} lakes...")

        # Process lakes in chunks
        results = []
        for i in range(0, total_lakes, lake_chunk_size):
            chunk_ids = object_ids[i:i + lake_chunk_size]
            logger.info(f"Processing chunk {i // lake_chunk_size + 1}/"
                        f"{(total_lakes + lake_chunk_size - 1) // lake_chunk_size} "
                        f"({len(chunk_ids)} lakes)...")

            # Calculate breakpoints for this chunk
            chunk_results = []
            for object_id in chunk_ids:
                try:
                    result = self.detector.calculate_break(filtered_dataset, object_id)
                    if not result.empty:
                        chunk_results.append(result)
                except Exception as e:
                    logger.warning(f"Error processing lake {object_id}: {e}")
                    continue

            if chunk_results:
                chunk_df = pd.concat(chunk_results)
                results.append(chunk_df)

                # Save intermediate results if requested
                if save_intermediate and output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    chunk_file = output_dir / f"{self.method}_results_chunk_{i // lake_chunk_size + 1}.csv"
                    chunk_df.to_csv(chunk_file)
                    logger.info(f"  Saved chunk results to {chunk_file}")

        # Combine all results
        if results:
            final_results = pd.concat(results, axis=0)
            logger.info(f"✅ Found breakpoints in {len(final_results)} lake-time combinations")
            return final_results
        else:
            logger.warning("No breakpoints found in the specified time range")
            return pd.DataFrame()

    def analyze_single_lake(
            self,
            dataset: DWDataset,
            object_id: str,
            start_date: Union[str, pd.Timestamp, datetime],
            end_date: Union[str, pd.Timestamp, datetime]
    ) -> pd.DataFrame:
        """Analyze breakpoints for a single lake over a time range.

        Parameters
        ----------
        dataset : DWDataset
            Dataset containing lake water-area data
        object_id : str
            Lake ID to analyze
        start_date : str or pd.Timestamp or datetime
            Start date for analysis
        end_date : str or pd.Timestamp or datetime
            End date for analysis

        Returns
        -------
        pd.DataFrame
            DataFrame containing breakpoint information
        """
        # Filter dataset to time range and specific lake
        filtered_ds = dataset.ds.sel(
            date=slice(pd.to_datetime(start_date), pd.to_datetime(end_date)),
            id_geohash=object_id
        )
        filtered_dataset = DWDataset(filtered_ds)

        return self.detector.calculate_break(filtered_dataset, object_id)


class BatchHistoricalAnalyzer:
    """Batch processor for running multiple historical analyses."""

    def __init__(self, output_dir: Union[str, Path]):
        """Initialize batch analyzer.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save all analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_multiple_analyses(
            self,
            dataset: DWDataset,
            analyses_config: List[dict]
    ) -> dict:
        """Run multiple historical analyses with different parameters.

        Parameters
        ----------
        dataset : DWDataset
            Dataset containing lake water-area data
        analyses_config : List[dict]
            List of analysis configurations, each containing:
            - name: str (output filename prefix)
            - method: str ("simple" or "beast")
            - start_date: str or Timestamp
            - end_date: str or Timestamp
            - threshold: float (optional)
            - window: int (optional)
            - break_threshold: float (optional)

        Returns
        -------
        dict
            Dictionary mapping analysis names to result DataFrames
        """
        results = {}

        for config in analyses_config:
            name = config['name']
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Running analysis: {name}")
            logger.info(f"{'=' * 60}")

            # Initialize analyzer for this configuration
            analyzer = HistoricalBreakpointAnalyzer(
                method=config.get('method', 'simple'),
                threshold=config.get('threshold'),
                window=config.get('window', 3),
                break_threshold=config.get('break_threshold', 0.5),
                **config.get('kwargs', {})
            )

            # Run analysis
            result = analyzer.analyze_time_range(
                dataset=dataset,
                start_date=config['start_date'],
                end_date=config['end_date'],
                object_ids=config.get('object_ids'),
                lake_chunk_size=config.get('lake_chunk_size', 1000),
                save_intermediate=True,
                output_dir=self.output_dir / name
            )

            # Save final results
            if not result.empty:
                output_file = self.output_dir / f"{name}_results.csv"
                result.to_csv(output_file)
                result.to_parquet(self.output_dir / f"{name}_results.parquet")
                logger.info(f"✅ Saved {name} results to {output_file}")
                results[name] = result
            else:
                logger.warning(f"No results for {name}")
                results[name] = pd.DataFrame()

        return results


# Example usage in your existing script
def precompute_historical_breakpoints(
        input_nc_file: str | Path,
        output_dir: str | Path,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        method: str = "simple",
        threshold: float = -0.25,
        window: int = 3,
        break_threshold: float = 0.5,
        lake_chunk_size: int = 1000,
        object_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """Precompute historical breakpoints for a time range.

    This function provides a simple interface similar to precompute_nrt_breakpoints.

    Parameters
    ----------
    input_nc_file : str or Path
        Path to the NetCDF file containing lake data
    output_dir : str or Path
        Directory to save results
    start_date : str or pd.Timestamp
        Start date for historical analysis
    end_date : str or pd.Timestamp
        End date for historical analysis
    method : str
        Breakpoint method: "simple" or "beast"
    threshold : float
        Threshold for SimpleBreakpoint (default: -0.25)
    window : int
        Window size for SimpleBreakpoint (default: 3)
    break_threshold : float
        Probability threshold for BeastBreakpoint (default: 0.5)
    lake_chunk_size : int
        Number of lakes to process in each chunk
    object_ids : List[str], optional
        Specific lake IDs to analyze

    Returns
    -------
    pd.DataFrame
        DataFrame containing breakpoint information
    """
    # Convert to Path objects
    input_nc_file = Path(input_nc_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    logger.info(f"Loading dataset from {input_nc_file}...")
    ds = xr.open_dataset(input_nc_file)
    dw_dataset = DWDataset(ds)

    # Initialize historical analyzer
    analyzer = HistoricalBreakpointAnalyzer(
        method=method,
        threshold=threshold,
        window=window,
        break_threshold=break_threshold
    )

    # Run analysis
    results = analyzer.analyze_time_range(
        dataset=dw_dataset,
        start_date=start_date,
        end_date=end_date,
        object_ids=object_ids,
        lake_chunk_size=lake_chunk_size,
        save_intermediate=True,
        output_dir=output_dir
    )

    # Save final results
    if not results.empty:
        # Save in multiple formats
        csv_file = output_dir / f"{method}_breakpoints_{start_date}_to_{end_date}.csv"
        parquet_file = output_dir / f"{method}_breakpoints_{start_date}_to_{end_date}.parquet"

        results.to_csv(csv_file)
        results.to_parquet(parquet_file)

        logger.info(f"✅ Results saved to {csv_file}")
        logger.info(f"✅ Parquet format saved to {parquet_file}")

        # Print summary statistics
        logger.info(f"\n📊 Summary Statistics:")
        logger.info(f"  - Total breakpoints found: {len(results)}")
        logger.info(f"  - Unique lakes with breaks: {results.index.nunique()}")

        if method == "simple":
            logger.info(f"  - Detection method: Simple (threshold={threshold}, window={window})")
        else:
            logger.info(f"  - Detection method: BEAST (probability threshold={break_threshold})")
            if 'proba_rbeast' in results.columns:
                logger.info(f"  - Mean break probability: {results['proba_rbeast'].mean():.3f}")

    return results