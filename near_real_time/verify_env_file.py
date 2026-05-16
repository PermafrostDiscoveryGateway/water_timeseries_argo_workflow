import sys
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv
import os

def load_environment():
    """
    Load environment variables with fallback priority:
    1. Command line argument (.env file path)
    2. Default ./.env file
    3. Kubernetes/OS environment variables (already present)
    """
    env_path = None

    # Priority 1: Command line argument for .env file
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        if Path(env_path).exists():
            load_dotenv(dotenv_path=env_path, override=False)  # Don't override existing env vars
            logger.info(f"Loaded environment from command line .env: {env_path}")
        else:
            logger.warning(f".env file not found at {env_path}, checking other sources")

    # Priority 2: Default .env file in current directory
    if not env_path or not Path(env_path).exists():
        default_env = Path.cwd() / ".env"
        if default_env.exists():
            load_dotenv(dotenv_path=default_env, override=False)
            logger.info(f"Loaded environment from default .env: {default_env}")
        else:
            logger.info("No .env file found, using Kubernetes/OS environment variables")

    # Priority 3: Kubernetes/OS environment variables are already in os.environ

    # Validate required variables (with helpful error messages)
    required_vars = [
        'output_dir',
        'project',
        'dynamic_world_dir',
        'vector_lake_file',
        'new_dynamic_world_data_dir'
    ]

    missing_vars = []
    for var in required_vars:
        if var not in os.environ:
            missing_vars.append(var)

    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        logger.info("Available environment variables: {list(os.environ.keys())}")
        raise EnvironmentError(error_msg)

    # Optional variables with defaults
    if 'dynamic_world_data_file' not in os.environ:
        logger.warning("dynamic_world_data_file not set, will use most recent file")

    # Log which source is providing each variable (debug)
    logger.debug("Environment configuration:")
    for var in required_vars:
        source = "K8s/OS" if var not in locals() else ".env"
        logger.debug(f"  {var} = {os.environ[var]} (source: {source})")


def main():
    load_environment()