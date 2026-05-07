#!/usr/bin/env python3
# run_analyses.py

import subprocess
import os
os.environ["ZARR_ASYNC"] = "0"
import sys
from pathlib import Path
import shlex
from datetime import datetime
import nest_asyncio
import sys
# import asyncio
# # Set the event loop policy for better compatibility
# if sys.platform == 'darwin':  # macOS
#     asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
# else:  # Linux (Kubernetes)
#     asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
#
# # Only use nest_asyncio if you're in a nested environment
# try:
#     loop = asyncio.get_running_loop()
#     nest_asyncio.apply()  # Only if already running in a loop
# except RuntimeError:
#     pass


# Generate config file list programmatically
def get_config_files():
    config_dir = Path("configs")

    # Option A: Specific list
    configs = [
        "configs/config1.yaml",
        "configs/config2.yaml",
    ]

    # Option B: Discover all YAML files
    # configs = list(config_dir.glob("config*.yaml"))

    # Option C: Generate based on dates
    # from your previous datetime logic
    # monthly_dates = generate_monthly_first_days(last_datetime)
    # configs = [f"configs/analysis_{dt.strftime('%Y%m')}.yaml" for dt in monthly_dates]

    return configs


def run_analyses(configs, parallel=False):
    if parallel:
        # Run in parallel
        processes = []
        for config in configs:
            cmd = ["uv", "run", "water-timeseries", "breakpoint-analysis", "--config-file", config]
            processes.append(subprocess.Popen(cmd))

        # Wait for all to complete
        for p in processes:
            p.wait()
    else:
        # Run sequentially
        for config in configs:
            result = subprocess.run(
                ["uv", "run", "water-timeseries", "breakpoint-analysis", "--config-file", config],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Error with {config}: {result.stderr}")
                sys.exit(1)


if __name__ == "__main__":
    # Add this before running the pipeline to inspect your data

    configfile_path = "/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/utils/test_config.yaml"

    cmd = [
        "uv", "run", "water-timeseries", "breakpoint-analysis",
        "--config-file", configfile_path
    ]

    # Print with proper shell quoting
    print(f"Running command: {shlex.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    print("Job finished")