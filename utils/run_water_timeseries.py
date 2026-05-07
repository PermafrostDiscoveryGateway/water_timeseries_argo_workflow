import argparse
import os
from loguru import logger
import toml

def load_config(config_path="/app/config/config.toml"):
    """Load configuration from TOML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = toml.load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Near Real Time Run")
    parser.add_argument("--config", help="Path to config file", default="/app/config/config.toml")
    args = parser.parse_args()
    config_path = "/app/config/config.toml"
    if args.config:
        config_path = args.config
    config = load_config(config_path=config_path)
    EE_PROJECT = config["project"]
    os.environ["EE_PROJECT"] = EE_PROJECT
    vector_dataset = config["vector_dataset"]
    path_to_dynamic_world_data = config["path_to_dynamic_world_data"]
    path_to_output = config["path_to_output"]
    years = config["years"]
    months = config["months"]
    bbox_west = config["bbox_west"]
    bbox_south = config["bbox_south"]
    bbox_north = config["bbox_north"]
    bbox_east = config["bbox_east"]
    chunksize = config["chunksize"]
    n_jobs = config["n_jobs"]
    print('went throgh the config')

    # uv run water-timeseries breakpoint-analysis --config-file configs/config.yaml



if __name__ == "__main__":
    main()