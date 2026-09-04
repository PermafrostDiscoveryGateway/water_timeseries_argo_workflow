"""
Splits the full global lake vector file into small per-region files, each
containing only the id_geohash + full lake polygon geometry for the lakes
whose centroid falls inside that region's bounding box.

The full vector file (`vector_lake_file`) is multiple GB, dominated by its
`geometry` column, but every per-region download job currently reads the
entire global file just to filter down to its own region (a few hundred
thousand lakes out of ~4 million). The download step needs the real polygon
geometry (Earth Engine reduces over each lake's actual shape), so the split
keeps full geometry - it just avoids repeatedly reading and filtering the
whole planet's worth of lakes in every region's job.

Run this once (or whenever the vector file / region boundaries change) to
produce `<region_lake_polygons_dir>/<REGION>_lake_polygons.parquet` files that
the per-region jobs can read instead of the multi-GB full vector file.
"""
import os
import sys
from pathlib import Path

import geopandas as gpd
from dotenv import load_dotenv
from loguru import logger

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.region_boundaries import get_region_boundaries


def compute_centroids(gdf):
    """Same representative-point logic as get_region_lakes(), computed once for the whole file."""
    geom_type = gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else None
    if geom_type in ['Polygon', 'MultiPolygon']:
        centroids = gdf.geometry.centroid
    elif geom_type == 'Point':
        centroids = gdf.geometry
    else:
        centroids = gdf.geometry.representative_point()
    return centroids.x, centroids.y


def generate_region_lake_polygon_files(vector_lake_file, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {vector_lake_file} (id_geohash + geometry only)...")
    gdf = gpd.read_parquet(vector_lake_file, columns=['id_geohash', 'geometry'])
    logger.info(f"Loaded {len(gdf):,} lakes")

    logger.info("Computing centroids...")
    x_coords, y_coords = compute_centroids(gdf)

    region_boundaries = get_region_boundaries()
    for region, bounds in region_boundaries.items():
        mask = (
            (x_coords >= bounds['X_MIN_START']) & (x_coords <= bounds['X_MIN_END']) &
            (y_coords >= bounds['Y_MIN_START']) & (y_coords <= bounds['Y_MIN_END'])
        )

        region_gdf = gdf.loc[mask]

        outfile = output_dir / f"{region}_lake_polygons.parquet"
        region_gdf.to_parquet(outfile)
        logger.info(f"  {region}: {len(region_gdf):,} lakes -> {outfile}")

    logger.info("Done.")


def main():
    env_path = None
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    vector_lake_file = os.environ['vector_lake_file']
    output_dir = os.environ['region_lake_polygons_dir']

    generate_region_lake_polygon_files(vector_lake_file, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
