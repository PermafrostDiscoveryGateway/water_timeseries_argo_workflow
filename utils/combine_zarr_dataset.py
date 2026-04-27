import xarray as xr
import zarr
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import shutil
import tempfile
from datetime import datetime
import logging


class ZarrDatasetCombinator:
    """
    A class for combining Zarr datasets, particularly useful for merging new data
    with existing datasets in a time-series context.
    """

    def __init__(self, base_path: Optional[Union[str, Path]] = None,
                 log_level: int = logging.INFO):
        """
        Initialize the combinator.

        Parameters:
        -----------
        base_path : str or Path, optional
            Path to the primary/base Zarr dataset
        log_level : int, optional
            Logging level (default: logging.INFO)
        """
        self.base_path = Path(base_path) if base_path else None
        self.logger = self._setup_logger(log_level)

    def _setup_logger(self, log_level):
        """Setup logging."""
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def combine_concatenate(self,
                            dataset1_path: Union[str, Path],
                            dataset2_path: Union[str, Path],
                            output_path: Union[str, Path],
                            dim: str = 'time',
                            combine_method: str = 'inner',
                            consolidate: bool = True,
                            remove_duplicates: bool = True) -> Dict[str, Any]:
        """
        Combine two Zarr datasets by concatenating along a dimension (typically time).
        This is the most common method for adding new time periods.

        Parameters:
        -----------
        dataset1_path : str or Path
            Path to first Zarr dataset (usually the base/old data)
        dataset2_path : str or Path
            Path to second Zarr dataset (usually the new data)
        output_path : str or Path
            Path where the combined dataset will be saved
        dim : str, default='time'
            Dimension to concatenate along
        combine_method : str, default='inner'
            How to handle variables: 'inner' (common variables only), 'outer' (all variables)
        consolidate : bool, default=True
            Whether to consolidate metadata after combining
        remove_duplicates : bool, default=True
            Whether to remove duplicate timesteps

        Returns:
        --------
        dict: Information about the combination
        """
        self.logger.info(f"Opening datasets: {dataset1_path} and {dataset2_path}")

        # Open both datasets
        ds1 = xr.open_zarr(dataset1_path, consolidated=True)
        ds2 = xr.open_zarr(dataset2_path, consolidated=True)

        # Check if the dimension exists in both
        if dim not in ds1.dims:
            raise ValueError(f"Dimension '{dim}' not found in first dataset")
        if dim not in ds2.dims:
            raise ValueError(f"Dimension '{dim}' not found in second dataset")

        # Check for overlapping time periods
        if remove_duplicates and dim == 'time':
            time1 = ds1.time.values
            time2 = ds2.time.values
            overlap = np.intersect1d(time1, time2)
            if len(overlap) > 0:
                self.logger.warning(f"Found {len(overlap)} overlapping timesteps. "
                                    f"Newer data will be prioritized.")
                # Remove overlapping times from the second dataset
                ds2 = ds2.sel(time=~np.isin(ds2.time.values, overlap))
                self.logger.info(f"After removing overlaps: {len(ds2.time)} timesteps remain")

        # Combine the datasets
        self.logger.info(f"Concatenating along dimension '{dim}'")
        combined = xr.concat([ds1, ds2], dim=dim, combine=combine_method)

        # Sort along the dimension if it's time
        if dim == 'time':
            combined = combined.sortby(dim)
            self.logger.info(f"Sorted by {dim}")

        # Save to output path
        self.logger.info(f"Saving combined dataset to {output_path}")
        combined.to_zarr(output_path, mode='w', consolidated=consolidate)

        # Get information about the result
        result_info = {
            'original_shape_ds1': dict(ds1.dims),
            'original_shape_ds2': dict(ds2.dims),
            'combined_shape': dict(combined.dims),
            'output_path': str(output_path),
            'dim_used': dim
        }

        # Close datasets
        ds1.close()
        ds2.close()
        combined.close()

        self.logger.info(f"Successfully combined datasets. Result shape: {result_info['combined_shape']}")
        return result_info

    def combine_merge(self,
                      dataset1_path: Union[str, Path],
                      dataset2_path: Union[str, Path],
                      output_path: Union[str, Path],
                      merge_method: str = 'inner') -> Dict[str, Any]:
        """
        Merge two Zarr datasets by variables (joining along coordinates).
        Useful when datasets have different variables but same time/space coordinates.

        Parameters:
        -----------
        dataset1_path, dataset2_path : str or Path
            Paths to the Zarr datasets
        output_path : str or Path
            Output path for merged dataset
        merge_method : str, default='inner'
            'inner' (only common coordinates), 'outer' (all coordinates)

        Returns:
        --------
        dict: Information about the merge
        """
        self.logger.info(f"Merging datasets: {dataset1_path} and {dataset2_path}")

        ds1 = xr.open_zarr(dataset1_path, consolidated=True)
        ds2 = xr.open_zarr(dataset2_path, consolidated=True)

        # Merge the datasets
        if merge_method == 'inner':
            merged = ds1.merge(ds2, compat='override')
        else:
            merged = xr.merge([ds1, ds2], compat='override')

        # Save
        self.logger.info(f"Saving merged dataset to {output_path}")
        merged.to_zarr(output_path, mode='w', consolidated=True)

        result_info = {
            'original_vars_ds1': list(ds1.data_vars.keys()),
            'original_vars_ds2': list(ds2.data_vars.keys()),
            'merged_vars': list(merged.data_vars.keys()),
            'output_path': str(output_path)
        }

        ds1.close()
        ds2.close()
        merged.close()

        self.logger.info(f"Successfully merged datasets")
        return result_info

    def append_to_existing(self,
                           new_data_path: Union[str, Path],
                           existing_data_path: Optional[Union[str, Path]] = None,
                           output_path: Optional[Union[str, Path]] = None,
                           dim: str = 'time',
                           in_place: bool = False) -> Dict[str, Any]:
        """
        Append new data to an existing Zarr dataset.
        This is optimized for incremental updates.

        Parameters:
        -----------
        new_data_path : str or Path
            Path to the new data to append
        existing_data_path : str or Path, optional
            Path to existing dataset (if None, uses self.base_path)
        output_path : str or Path, optional
            Output path (if None and not in_place, generates new name)
        dim : str, default='time'
            Dimension to append along
        in_place : bool, default=False
            If True, overwrite the existing dataset (use carefully!)

        Returns:
        --------
        dict: Information about the append operation
        """
        if existing_data_path is None:
            if self.base_path is None:
                raise ValueError("Must provide existing_data_path or set base_path")
            existing_data_path = self.base_path

        existing_path = Path(existing_data_path)

        if output_path is None and not in_place:
            # Generate output path with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = existing_path.parent / f"{existing_path.stem}_combined_{timestamp}.zarr"
        elif in_place:
            output_path = existing_path
            # Create backup
            backup_path = existing_path.parent / f"{existing_path.stem}_backup_{timestamp}.zarr"
            self.logger.info(f"Creating backup at {backup_path}")
            shutil.copytree(existing_path, backup_path)

        # Perform the combination
        result = self.combine_concatenate(
            dataset1_path=existing_path,
            dataset2_path=new_data_path,
            output_path=output_path,
            dim=dim,
            remove_duplicates=True
        )

        # If in_place, we've already overwritten
        if not in_place:
            result['backup_created'] = False
        else:
            result['backup_created'] = True
            result['backup_path'] = str(backup_path)

        return result

    def combine_multiple(self,
                         dataset_paths: List[Union[str, Path]],
                         output_path: Union[str, Path],
                         dim: str = 'time',
                         chunk_size: Optional[int] = 100) -> Dict[str, Any]:
        """
        Combine multiple Zarr datasets efficiently.

        Parameters:
        -----------
        dataset_paths : list of str/Path
            List of paths to Zarr datasets to combine
        output_path : str or Path
            Output path
        dim : str, default='time'
            Dimension to concatenate along
        chunk_size : int, optional
            Process in chunks to save memory (if None, process all at once)

        Returns:
        --------
        dict: Information about the combination
        """
        self.logger.info(f"Combining {len(dataset_paths)} datasets")

        if chunk_size:
            # Process in chunks
            combined = None
            for i in range(0, len(dataset_paths), chunk_size):
                chunk_paths = dataset_paths[i:i + chunk_size]
                self.logger.info(f"Processing chunk {i // chunk_size + 1} of {len(dataset_paths) // chunk_size + 1}")

                chunk_ds = xr.open_mfdataset(
                    chunk_paths,
                    combine='nested',
                    concat_dim=dim,
                    engine='zarr'
                )

                if combined is None:
                    combined = chunk_ds
                else:
                    combined = xr.concat([combined, chunk_ds], dim=dim)
        else:
            # Process all at once
            combined = xr.open_mfdataset(
                dataset_paths,
                combine='nested',
                concat_dim=dim,
                engine='zarr'
            )

        # Sort by dimension
        if dim == 'time':
            combined = combined.sortby(dim)

        # Save
        self.logger.info(f"Saving combined dataset to {output_path}")
        combined.to_zarr(output_path, mode='w', consolidated=True)

        result_info = {
            'num_datasets': len(dataset_paths),
            'combined_shape': dict(combined.dims),
            'output_path': str(output_path)
        }

        combined.close()

        return result_info

    def check_for_updates(self,
                          current_dataset_path: Union[str, Path],
                          new_dataset_path: Union[str, Path],
                          dim: str = 'time') -> Dict[str, Any]:
        """
        Check what new data would be added by combining two datasets.

        Parameters:
        -----------
        current_dataset_path : str or Path
            Path to current dataset
        new_dataset_path : str or Path
            Path to new dataset with potential updates
        dim : str, default='time'
            Dimension to check for new values

        Returns:
        --------
        dict: Information about new data available
        """
        ds_current = xr.open_zarr(current_dataset_path, consolidated=True)
        ds_new = xr.open_zarr(new_dataset_path, consolidated=True)

        if dim not in ds_current.dims or dim not in ds_new.dims:
            self.logger.warning(f"Dimension '{dim}' not found in one of the datasets")
            return {'has_new_data': False, 'error': f"Dimension '{dim}' missing"}

        # Get the dimension values
        current_vals = ds_current[dim].values
        new_vals = ds_new[dim].values

        # Find new values
        if dim == 'time':
            # Handle datetime comparison
            try:
                new_mask = ~np.isin(new_vals, current_vals)
                new_values = new_vals[new_mask]
            except:
                # Convert to strings for comparison if datetime comparison fails
                new_mask = ~np.isin(new_vals.astype(str), current_vals.astype(str))
                new_values = new_vals[new_mask]
        else:
            new_mask = ~np.isin(new_vals, current_vals)
            new_values = new_vals[new_mask]

        result = {
            'has_new_data': len(new_values) > 0,
            'num_new_values': len(new_values),
            'new_values': new_values[:10] if len(new_values) > 10 else new_values,  # First 10
            'total_in_new': len(new_vals),
            'dimension': dim
        }

        if len(new_values) > 10:
            result['message'] = f"Showing first 10 of {len(new_values)} new values"

        ds_current.close()
        ds_new.close()

        return result


# Convenience functions for simple operations
def quick_combine_zarr(dataset1_path: Union[str, Path],
                       dataset2_path: Union[str, Path],
                       output_path: Union[str, Path],
                       dim: str = 'time') -> Dict[str, Any]:
    """
    Quick one-shot function to combine two Zarr datasets.
    """
    combinator = ZarrDatasetCombinator()
    return combinator.combine_concatenate(dataset1_path, dataset2_path, output_path, dim)


def append_new_data(new_data_path: Union[str, Path],
                    existing_data_path: Union[str, Path],
                    dim: str = 'time') -> Dict[str, Any]:
    """
    Quick function to append new data to existing dataset.
    Creates a new combined dataset with timestamp.
    """
    combinator = ZarrDatasetCombinator()
    return combinator.append_to_existing(new_data_path, existing_data_path, dim=dim)


# Example usage
if __name__ == "__main__":
    # Initialize the combinator
    combinator = ZarrDatasetCombinator(base_path="path/to/base_data.zarr")

    # Example 1: Simple combine of two datasets
    result = combinator.combine_concatenate(
        dataset1_path="old_data.zarr",
        dataset2_path="new_data.zarr",
        output_path="combined_data.zarr",
        dim='time'
    )
    print(f"Combined shape: {result['combined_shape']}")

    # Example 2: Append new data to existing dataset
    result = combinator.append_to_existing(
        new_data_path="new_download.zarr",
        existing_data_path="existing_data.zarr",
        output_path="updated_data.zarr"
    )

    # Example 3: Check what new data would be added
    update_summary = combinator.check_for_updates(
        current_dataset_path="existing_data.zarr",
        new_dataset_path="new_download.zarr"
    )
    print(f"Found {update_summary['num_new_values']} new timesteps")

    # Example 4: Combine multiple files at once
    all_files = ["data_2020.zarr", "data_2021.zarr", "data_2022.zarr"]
    result = combinator.combine_multiple(
        dataset_paths=all_files,
        output_path="all_years.zarr",
        chunk_size=2  # Process 2 at a time to save memory
    )

    # Quick one-liner for simple cases
    result = quick_combine_zarr("dataset1.zarr", "dataset2.zarr", "combined.zarr")