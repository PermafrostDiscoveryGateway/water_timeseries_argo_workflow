#!/usr/bin/env python3
# run_analyses.py

import subprocess
import sys
from pathlib import Path
from datetime import datetime


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
    config_files = get_config_files()
    print(f"Running {len(config_files)} analyses")
    run_analyses(config_files, parallel=False)