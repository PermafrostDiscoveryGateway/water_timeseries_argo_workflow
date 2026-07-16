from typing import List
import xarray as xr
from pathlib import Path
from loguru import logger
import numpy as np

def compare_netcdf_files(input_files: List[str], output_files: List[str]) -> dict:
    """
    Compare one or more input NetCDF files against one or more output NetCDF
    files to determine whether all data present in the inputs is also present
    in the outputs.

    Args:
        input_files: List of paths to input NetCDF file(s)
        output_files: List of paths to output NetCDF file(s)

    Returns:
        dict: JSON-serializable results, including 'all_data_present': bool
    """
    logger.debug(f"Comparing inputs to outputs")

    def print_section(title, char='='):
        print(f"\n{char * 80}")
        print(f"{title}")
        print(f"{char * 80}")

    result = {
        'input_files': [str(f) for f in input_files],
        'output_files': [str(f) for f in output_files],
        'all_data_present': False,
        'missing_variables': [],
        'variables': {},
        'errors': [],
    }

    # ========== OPEN AND COMBINE FILES ==========
    def open_combined(files, label):
        try:
            if len(files) == 1:
                return xr.open_dataset(files[0])
            return xr.open_mfdataset(files, combine='by_coords')
        except Exception as e:
            msg = f"Error opening {label} files: {e}"
            logger.error(msg)
            result['errors'].append(msg)
            return None

    input_ds = open_combined(input_files, 'input')
    output_ds = open_combined(output_files, 'output')

    if input_ds is None or output_ds is None:
        return result

    try:
        print_section("BASIC FILE INFORMATION")
        print(f"Input files ({len(input_files)}): {[Path(f).name for f in input_files]}")
        print(f"Output files ({len(output_files)}): {[Path(f).name for f in output_files]}")
        print(f"Input dimensions: {dict(input_ds.dims)}")
        print(f"Output dimensions: {dict(output_ds.dims)}")
        print(f"Input variables: {list(input_ds.data_vars)}")
        print(f"Output variables: {list(output_ds.data_vars)}")

        # ========== VARIABLE PRESENCE ==========
        print_section("VARIABLE PRESENCE CHECK")
        missing_variables = [v for v in input_ds.data_vars if v not in output_ds.data_vars]
        result['missing_variables'] = missing_variables
        if missing_variables:
            print(f"  Variables missing from output entirely: {missing_variables}")
        else:
            print("  All input variables are present in output.")

        # ========== PER-VARIABLE COMPARISON ==========
        print_section("PER-VARIABLE DATA COVERAGE")
        overall_ok = not missing_variables

        for var_name in input_ds.data_vars:
            var_result = {
                'present_in_output': var_name in output_ds.data_vars,
                'input_count': 0,
                'output_count': 0,
                'missing_coord_points': {},
                'value_mismatches': None,
                'all_present': False,
            }

            if var_name not in output_ds.data_vars:
                result['variables'][var_name] = var_result
                overall_ok = False
                continue

            in_var = input_ds[var_name]
            out_var = output_ds[var_name]

            in_valid = int(in_var.notnull().sum()) if np.issubdtype(in_var.dtype, np.floating) else int(in_var.size)
            var_result['input_count'] = in_valid

            # Check that every dimension/coordinate value in the input exists in the output
            missing_coords = {}
            for dim in in_var.dims:
                if dim in input_ds.coords and dim in output_ds.coords:
                    in_coord_vals = input_ds.coords[dim].values
                    out_coord_vals = set(np.atleast_1d(output_ds.coords[dim].values))
                    missing_vals = [v for v in in_coord_vals if v not in out_coord_vals]
                    if missing_vals:
                        missing_coords[dim] = len(missing_vals)
            var_result['missing_coord_points'] = missing_coords

            # Align on shared coordinates and compare overlapping values
            try:
                aligned_in, aligned_out = xr.align(in_var, out_var, join='inner')
                if aligned_in.size == 0:
                    print(f"  {var_name}: no overlapping coordinates between input and output")
                    var_result['value_mismatches'] = 'no_overlap'
                    overall_ok = False
                else:
                    if np.issubdtype(aligned_in.dtype, np.floating):
                        both_nan = aligned_in.isnull() & aligned_out.isnull()
                        close = xr.apply_ufunc(np.isclose, aligned_in, aligned_out,
                                                kwargs={'equal_nan': True})
                        mismatches = int((~close & ~both_nan).sum())
                    else:
                        mismatches = int((aligned_in != aligned_out).sum())
                    var_result['value_mismatches'] = mismatches
                    if mismatches > 0:
                        print(f"  {var_name}: {mismatches:,} mismatched overlapping value(s)")
                        overall_ok = False
            except Exception as e:
                var_result['value_mismatches'] = f'error: {e}'
                overall_ok = False

            out_valid = int(out_var.notnull().sum()) if np.issubdtype(out_var.dtype, np.floating) else int(out_var.size)
            var_result['output_count'] = out_valid

            var_result['all_present'] = (
                not missing_coords
                and var_result['value_mismatches'] == 0
                and out_valid >= in_valid
            )
            if not var_result['all_present']:
                overall_ok = False

            print(f"  {var_name}: input_count={in_valid:,}, output_count={out_valid:,}, "
                  f"missing_coord_points={missing_coords}, "
                  f"all_present={var_result['all_present']}")

            result['variables'][var_name] = var_result

        result['all_data_present'] = overall_ok

        # ========== SUMMARY ==========
        print_section("SUMMARY")
        print(f"All input data present in output: {result['all_data_present']}")
        if missing_variables:
            print(f"Missing variables: {missing_variables}")
        for var_name, var_result in result['variables'].items():
            if not var_result['all_present']:
                print(f"  {var_name}: NOT fully present -> {var_result}")

    finally:
        input_ds.close()
        output_ds.close()

    print_section("COMPARISON COMPLETE", '=')

    return result