import os
import toml
from loguru import logger
from water_timeseries.downloader import EarthEngineDownloader
import nest_asyncio

nest_asyncio.apply()


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
    # Load TOML config
    config = load_config()

    # Override with environment variables (if present)
    ee_project = os.environ.get("EE_PROJECT") or config.get("ee", {}).get("project", "pdg-project-406720")
    vector_dataset = os.environ.get("VECTOR_DATASET") or config.get("vector_dataset", "")
    save_to_file = os.environ.get("SAVE_TO") or config.get("output", {}).get("save_path", "")

    # Get years/months from environment or config with defaults
    years_str = os.environ.get("YEARS", "")
    months_str = os.environ.get("MONTHS", "")

    if years_str:
        years = [int(y.strip()) for y in years_str.split(",")]
    else:
        years = config.get("defaults", {}).get("years", [2024])

    if months_str:
        months = [int(m.strip()) for m in months_str.split(",")]
    else:
        months = config.get("defaults", {}).get("months", [7, 8])

    # Command-line arguments override everything (if passed via argparse)
    import argparse
    parser = argparse.ArgumentParser(description="Download Dynamic World data")
    parser.add_argument("--config", help="Path to config file", default="/app/config/config.toml")
    parser.add_argument("--years", help="Years to download (comma-separated)")
    parser.add_argument("--months", help="Months to download (comma-separated)")
    parser.add_argument("--vector-dataset", help="Path to vector dataset")
    parser.add_argument("--save-to", help="Path to save output")
    parser.add_argument("--ee-project", help="Earth Engine project ID")

    args = parser.parse_args()

    # Command-line args override everything
    if args.config and args.config != "/app/config/config.toml":
        with open(args.config, 'r') as f:
            config = toml.load(f)

    final_years = [int(y) for y in args.years.split(",")] if args.years else years
    final_months = [int(m) for m in args.months.split(",")] if args.months else months
    final_vector_dataset = args.vector_dataset or vector_dataset
    final_save_to = args.save_to or save_to_file
    final_ee_project = args.ee_project or ee_project

    # Set environment variable for EE
    os.environ["EE_PROJECT"] = final_ee_project

    logger.info(f"Using EE Project: {final_ee_project}")
    logger.info(f"Years: {final_years}, Months: {final_months}")
    logger.info(f"Vector dataset: {final_vector_dataset}")
    logger.info(f"Save to: {final_save_to}")

    # Initialize downloader
    dl = EarthEngineDownloader(ee_auth=True, logger=logger)

    # Download data
    ds = dl.download_dw_monthly(
        vector_dataset=final_vector_dataset,
        name_attribute="id_geohash",
        years=final_years,
        months=final_months,
        save_to_file=final_save_to,
    )

    logger.info(f"Successfully downloaded data to {final_save_to}")


if __name__ == "__main__":
    main()