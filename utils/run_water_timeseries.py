import argparse

def main():
    parser = argparse.ArgumentParser(description="Near Real Time Run")
    parser.add_argument("--config", help="Path to config file", default="/app/config/config.toml")
    args = parser.parse_args()

if __name__ == "__main__":
    main()