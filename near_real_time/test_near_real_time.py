import subprocess
import sys
import os
import test
import download_region
from loguru import logger

def load_env_file(env_path):
    """Load environment variables from a .env file"""
    env_vars = {}
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                env_vars[key] = value
    return env_vars


import subprocess
import sys
import os


def load_env_file(env_path):
    """Load environment variables from a .env file"""
    env_vars = {}
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                env_vars[key] = value
    return env_vars

# example of how to run a script using an env file
def run_scripts_with_env():
    # Load environment variables from test.env
    path_to_env = os.path.join(os.getcwd(), 'near_real_time', '.env')
    env_vars = load_env_file(path_to_env)

    # Merge with current environment (so existing vars are preserved)
    env = os.environ.copy()
    env.update(env_vars)
    logger.info(f"=== ENV VARS: {env_vars} for testing")

    # logger.debug("Running download region for region TEST")
    # download_region.main()

    logger.debug("Running download region for region EURASIA3")
    env['region_name'] = 'EURASIA3'
    download_region.main()





if  __name__ == "__main__":
    run_scripts_with_env()