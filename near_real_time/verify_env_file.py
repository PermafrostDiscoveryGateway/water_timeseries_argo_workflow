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
        'new_dynamic_world_data_dir',
        'base_dir',
        'split_vector_dataset_dir',
        'dynamic_world_data_file',
        'split_new_dynamic_world_data_dir'
    ]

    missing_vars = []
    for var in required_vars:
        if var not in os.environ:
            missing_vars.append(var)

    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        logger.info(f"Available environment variables: {list(os.environ.keys())}")
        raise EnvironmentError(error_msg)

    # Log which source is providing each variable (debug)
    logger.debug("Environment configuration:")
    for var in required_vars:
        logger.debug(f"  {var} = {os.environ[var]}")


def validate_paths():
    """
    Validate that environment variables containing paths exist and are accessible.
    Returns True if all paths are valid, False otherwise.
    """
    # List of environment variables that represent directory paths
    dir_path_vars = [
        'base_dir',
        'dynamic_world_dir',
        'new_dynamic_world_data_dir',
        'output_dir',
        'split_vector_dataset_dir',
        'split_new_dynamic_world_data_dir'
    ]

    # List of environment variables that represent file paths
    file_path_vars = [
        'vector_lake_file',
        'dynamic_world_data_file'
    ]

    all_valid = True

    # Validate directory paths
    for var in dir_path_vars:
        if var in os.environ:
            path = Path(os.environ[var])
            if path.exists():
                if path.is_dir():
                    logger.info(f"✓ {var} = {path} (directory exists)")
                else:
                    logger.error(f"✗ {var} = {path} exists but is NOT a directory")
                    all_valid = False
            else:
                logger.error(f"✗ {var} = {path} does NOT exist")
                all_valid = False

    # Validate file paths
    for var in file_path_vars:
        if var in os.environ:
            path = Path(os.environ[var])
            if path.exists():
                if path.is_file():
                    logger.info(f"✓ {var} = {path} (file exists)")
                else:
                    logger.error(f"✗ {var} = {path} exists but is NOT a file")
                    all_valid = False
            else:
                # For files, also check if parent directory exists
                if path.parent.exists():
                    logger.warning(
                        f"⚠ {var} = {path} does not exist yet, but parent directory is valid (file may be created later)")
                else:
                    logger.error(f"✗ {var} = {path} does NOT exist and parent directory is missing")
                    all_valid = False

    # Additional check for project (non-path variable)
    if 'project' in os.environ:
        logger.info(f"✓ project = {os.environ['project']}")

    return all_valid


def main():
    """Main function to load environment and validate paths"""
    logger.info("Starting environment validation...")

    # Load environment variables
    try:
        load_environment()
        logger.info("✓ Environment variables loaded successfully")
    except EnvironmentError as e:
        logger.error(f"Failed to load environment variables: {e}")
        sys.exit(1)

    # Validate paths
    logger.info("\nValidating paths...")
    if validate_paths():
        logger.info("\n✓ All environment variables and paths are valid!")
        return True
    else:
        logger.error("\n✗ Path validation failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    # Run validation
    valid = main()

    # Exit with appropriate code
    sys.exit(0 if valid else 1)