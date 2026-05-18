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
                    method=kwargs.get('method_name', "median"),
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

    def analyze_dates(
            self,
            dataset: DWDataset,
            analysis_dates: List[Union[str, pd.Timestamp, datetime]],
            object_ids: Optional[List[str]] = None,
            lake_chunk_size: int = 1000,
            save_intermediate: bool = True,
            output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """Analyze breakpoints for specific dates across all lakes.

        For each analysis date, the detector will identify breakpoints
        using data up to that date.

        Parameters
        ----------
        dataset : DWDataset
            Dataset containing lake water-area data
        analysis_dates : List[str or pd.Timestamp or datetime]
            List of dates to analyze breakpoints for
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
            DataFrame containing breakpoint information for each date
        """
        # Convert all dates to pandas Timestamp
        analysis_dates = [pd.to_datetime(date) for date in analysis_dates]

        logger.info(f"Analyzing breakpoints for {len(analysis_dates)} dates: {analysis_dates}")
        logger.info(f"Using method: {self.method}")

        # Determine which lakes to process
        if object_ids is None:
            object_ids = dataset.object_ids_
        else:
            # Filter to only include lakes that exist in dataset
            object_ids = [oid for oid in object_ids if oid in dataset.object_ids_]

        total_lakes = len(object_ids)
        logger.info(f"Processing {total_lakes} lakes...")

        all_results = []

        # Process each analysis date
        for date_idx, analysis_date in enumerate(analysis_dates, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing analysis date {date_idx}/{len(analysis_dates)}: {analysis_date}")
            logger.info(f"{'=' * 60}")

            # Filter dataset to data up to analysis date
            filtered_ds = dataset.ds.sel(date=slice(None, analysis_date))
            filtered_dataset = DWDataset(filtered_ds)

            # Process lakes in chunks for this date
            date_results = []
            for i in range(0, total_lakes, lake_chunk_size):
                chunk_ids = object_ids[i:i + lake_chunk_size]
                logger.info(f"Processing chunk {i // lake_chunk_size + 1}/"
                            f"{(total_lakes + lake_chunk_size - 1) // lake_chunk_size} "
                            f"({len(chunk_ids)} lakes) for date {analysis_date}...")

                # Calculate breakpoints for this chunk
                chunk_results = []
                for object_id in chunk_ids:
                    try:
                        result = self.detector.calculate_break(filtered_dataset, object_id)
                        if not result.empty:
                            # Add analysis date to results
                            result['analysis_date'] = analysis_date
                            chunk_results.append(result)
                    except Exception as e:
                        logger.warning(f"Error processing lake {object_id} for date {analysis_date}: {e}")
                        continue

                if chunk_results:
                    chunk_df = pd.concat(chunk_results)
                    date_results.append(chunk_df)

                    # Save intermediate results if requested
                    if save_intermediate and output_dir:
                        output_dir = Path(output_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        chunk_file = output_dir / f"{self.method}_results_date_{analysis_date.strftime('%Y%m%d')}_chunk_{i // lake_chunk_size + 1}.csv"
                        chunk_df.to_csv(chunk_file)
                        logger.info(f"  Saved chunk results to {chunk_file}")

            # Combine results for this date
            if date_results:
                date_combined = pd.concat(date_results)
                all_results.append(date_combined)
                logger.info(f"✅ Found {len(date_combined)} breakpoints for date {analysis_date}")
            else:
                logger.warning(f"No breakpoints found for date {analysis_date}")

        # Combine all results across dates
        if all_results:
            final_results = pd.concat(all_results, axis=0)
            logger.info(f"\n✅ Total breakpoints found across all dates: {len(final_results)}")
            return final_results
        else:
            logger.warning("No breakpoints found for any analysis date")
            return pd.DataFrame()

    def analyze_time_range(
            self,
            dataset: DWDataset,
            start_date: Union[str, pd.Timestamp, datetime],
            end_date: Union[str, pd.Timestamp, datetime],
            step_months: int = 1,
            object_ids: Optional[List[str]] = None,
            lake_chunk_size: int = 1000,
            save_intermediate: bool = True,
            output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """Analyze breakpoints over a historical time range with regular intervals.

        This is a convenience method that generates analysis dates at regular
        intervals and calls analyze_dates().

        Parameters
        ----------
        dataset : DWDataset
            Dataset containing lake water-area data
        start_date : str or pd.Timestamp or datetime
            Start date for historical analysis
        end_date : str or pd.Timestamp or datetime
            End date for historical analysis
        step_months : int
            Number of months between analysis dates (default: 1)
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
            DataFrame containing breakpoint information for each analysis date
        """
        # Convert dates to pandas Timestamp
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Generate analysis dates at regular intervals
        analysis_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq=f'{step_months}MS'  # Month start frequency
        )

        logger.info(f"Generated {len(analysis_dates)} analysis dates from {start_date} to {end_date}")

        return self.analyze_dates(
            dataset=dataset,
            analysis_dates=analysis_dates,
            object_ids=object_ids,
            lake_chunk_size=lake_chunk_size,
            save_intermediate=save_intermediate,
            output_dir=output_dir
        )

    def analyze_single_lake(
            self,
            dataset: DWDataset,
            object_id: str,
            analysis_dates: List[Union[str, pd.Timestamp, datetime]]
    ) -> pd.DataFrame:
        """Analyze breakpoints for a single lake at specific dates.

        Parameters
        ----------
        dataset : DWDataset
            Dataset containing lake water-area data
        object_id : str
            Lake ID to analyze
        analysis_dates : List[str or pd.Timestamp or datetime]
            List of dates to analyze breakpoints for

        Returns
        -------
        pd.DataFrame
            DataFrame containing breakpoint information for each analysis date
        """
        # Convert all dates to pandas Timestamp
        analysis_dates = [pd.to_datetime(date) for date in analysis_dates]

        results = []
        for analysis_date in analysis_dates:
            # Filter dataset to data up to analysis date and specific lake
            filtered_ds = dataset.ds.sel(
                date=slice(None, analysis_date),
                id_geohash=object_id
            )
            filtered_dataset = DWDataset(filtered_ds)

            result = self.detector.calculate_break(filtered_dataset, object_id)
            if not result.empty:
                result['analysis_date'] = analysis_date
                results.append(result)

        if results:
            return pd.concat(results)
        else:
            return pd.DataFrame()


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
            - analysis_dates: List[str] (dates to analyze)
            OR
            - start_date: str
            - end_date: str
            - step_months: int (optional, default 1)
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

            # Run analysis - either with explicit dates or time range
            if 'analysis_dates' in config:
                result = analyzer.analyze_dates(
                    dataset=dataset,
                    analysis_dates=config['analysis_dates'],
                    object_ids=config.get('object_ids'),
                    lake_chunk_size=config.get('lake_chunk_size', 1000),
                    save_intermediate=True,
                    output_dir=self.output_dir / name
                )
            else:
                result = analyzer.analyze_time_range(
                    dataset=dataset,
                    start_date=config['start_date'],
                    end_date=config['end_date'],
                    step_months=config.get('step_months', 1),
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
        analysis_dates: List[Union[str, pd.Timestamp, datetime]],
        method: str = "simple",
        threshold: float = -0.25,
        window: int = 3,
        break_threshold: float = 0.5,
        lake_chunk_size: int = 1000,
        object_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """Precompute historical breakpoints for specific dates.

    This function provides a simple interface similar to precompute_nrt_breakpoints.

    Parameters
    ----------
    input_nc_file : str or Path
        Path to the NetCDF file containing lake data
    output_dir : str or Path
        Directory to save results
    analysis_dates : List[str or pd.Timestamp or datetime]
        List of dates to analyze breakpoints for
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
    results = analyzer.analyze_dates(
        dataset=dw_dataset,
        analysis_dates=analysis_dates,
        object_ids=object_ids,
        lake_chunk_size=lake_chunk_size,
        save_intermediate=True,
        output_dir=output_dir
    )

    # Save final results
    if not results.empty:
        # Create date range string for filename
        date_str = f"{pd.to_datetime(analysis_dates[0]).strftime('%Y%m%d')}_to_{pd.to_datetime(analysis_dates[-1]).strftime('%Y%m%d')}"

        # Save in multiple formats
        csv_file = output_dir / f"{method}_breakpoints_{date_str}.csv"
        parquet_file = output_dir / f"{method}_breakpoints_{date_str}.parquet"

        results.to_csv(csv_file)
        results.to_parquet(parquet_file)

        logger.info(f"✅ Results saved to {csv_file}")
        logger.info(f"✅ Parquet format saved to {parquet_file}")

        # Print summary statistics
        logger.info(f"\n📊 Summary Statistics:")
        logger.info(f"  - Total breakpoints found: {len(results)}")
        logger.info(f"  - Unique lakes with breaks: {results.index.nunique()}")
        logger.info(f"  - Analysis dates: {results['analysis_date'].nunique()} unique dates")

        if method == "simple":
            logger.info(f"  - Detection method: Simple (threshold={threshold}, window={window})")
        else:
            logger.info(f"  - Detection method: BEAST (probability threshold={break_threshold})")
            if 'proba_rbeast' in results.columns:
                logger.info(f"  - Mean break probability: {results['proba_rbeast'].mean():.3f}")

    return results