import os
import pandas as pd
import xarray as xr
import numpy as np
from dotenv import load_dotenv
from water_timeseries.utils.io import load_vector_dataset


def analyze_vector_file(vector_file):
    """
    Analyze vector file structure and memory usage
    """
    print("=" * 70)
    print("ANALYZING VECTOR FILE")
    print("=" * 70)

    # Load vector file
    print(f"\nLoading vector file: {vector_file}")
    gdf = load_vector_dataset(vector_file)

    # Basic info
    print(f"\n--- Basic Information ---")
    print(f"Total records: {len(gdf):,}")
    print(f"Columns: {list(gdf.columns)}")

    # Check for id_geohash column
    id_column = 'id_geohash'
    if id_column not in gdf.columns:
        print(f"\n⚠️ Warning: '{id_column}' column not found!")
        print(f"Available columns: {list(gdf.columns)}")
        return None

    # ID analysis
    print(f"\n--- ID Analysis ---")
    ids = gdf[id_column].values
    print(f"Unique IDs: {len(np.unique(ids)):,}")
    print(f"Duplicate IDs: {len(ids) - len(np.unique(ids)):,}")

    # Sample IDs
    print(f"\n--- Sample IDs (first 10) ---")
    for i, lake_id in enumerate(ids[:10]):
        print(f"  {i + 1}: {lake_id}")

    # Geometry analysis
    print(f"\n--- Geometry Analysis ---")
    if 'geometry' in gdf.columns:
        geom_types = gdf.geometry.geom_type.value_counts()
        print(f"Geometry types:")
        for geom_type, count in geom_types.items():
            print(f"  {geom_type}: {count:,}")

        # Check geometry complexity
        print(f"\n--- Geometry Complexity ---")
        vertex_counts = []
        for geom in gdf.geometry[:1000]:  # Sample first 1000
            try:
                if geom.geom_type == 'Polygon':
                    vertex_counts.append(len(geom.exterior.coords))
                elif geom.geom_type == 'MultiPolygon':
                    total = sum(len(poly.exterior.coords) for poly in geom.geoms)
                    vertex_counts.append(total)
            except:
                pass

        if vertex_counts:
            print(f"Average vertices per geometry (sample 1000): {np.mean(vertex_counts):.0f}")
            print(f"Max vertices: {np.max(vertex_counts):,}")
            print(f"Min vertices: {np.min(vertex_counts)}")

    # Memory usage
    print(f"\n--- Memory Usage ---")
    memory_mb = gdf.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"DataFrame memory: {memory_mb:.1f} MB")

    # Estimate Earth Engine request size
    print(f"\n--- Earth Engine Estimation ---")
    print(f"With 1 lake: ~{memory_mb / len(gdf) * 1:.1f} MB")
    print(f"With 1,000 lakes: ~{memory_mb / len(gdf) * 1000:.1f} MB")
    print(f"With 10,000 lakes: ~{memory_mb / len(gdf) * 10000:.1f} MB")
    print(f"With 50,000 lakes: ~{memory_mb / len(gdf) * 50000:.1f} MB")

    return gdf


def test_split_strategies(gdf, id_column='id_geohash', sample_size=10000):
    """
    Test different splitting strategies
    """
    print("\n" + "=" * 70)
    print("TESTING SPLIT STRATEGIES")
    print("=" * 70)

    ids = gdf[id_column].values
    unique_ids = np.unique(ids)

    # Strategy 1: Simple chunking
    print(f"\n--- Strategy 1: Simple Chunking ---")
    chunk_sizes = [1000, 5000, 10000, 25000, 50000]

    for chunk_size in chunk_sizes:
        n_chunks = len(unique_ids) // chunk_size + (1 if len(unique_ids) % chunk_size else 0)
        print(f"  Chunk size {chunk_size:,}: {n_chunks} chunks")

    # Strategy 2: Sample a chunk and check geometry size
    print(f"\n--- Strategy 2: Sample Chunk Geometry Analysis ---")
    sample_ids = unique_ids[:sample_size]
    sample_gdf = gdf[gdf[id_column].isin(sample_ids)]

    sample_memory_mb = sample_gdf.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"  Sample size: {len(sample_ids):,} lakes")
    print(f"  Sample memory: {sample_memory_mb:.1f} MB")

    # Estimate safe chunk size based on 8GB memory limit
    safe_memory_mb = 8000  # 8GB limit for safe operation
    safe_chunk_size = int((safe_memory_mb / sample_memory_mb) * len(sample_ids))
    print(f"\n  Recommended safe chunk size: {safe_chunk_size:,} lakes")
    print(f"  (Based on 8GB memory limit, your current sample uses {sample_memory_mb:.1f} MB)")

    # Strategy 3: Check for natural groupings (if available)
    print(f"\n--- Strategy 3: Natural Groupings ---")
    if 'continent' in gdf.columns:
        print(f"  Continent distribution:")
        for continent, count in gdf['continent'].value_counts().head(10).items():
            print(f"    {continent}: {count:,}")

    if 'country' in gdf.columns:
        print(f"\n  Top 10 countries by lake count:")
        for country, count in gdf['country'].value_counts().head(10).items():
            print(f"    {country}: {count:,}")

    return safe_chunk_size


def compare_with_dynamic_world(dynamic_world_dir, vector_gdf, id_column='id_geohash'):
    """
    Compare vector file IDs with Dynamic World files
    """
    print("\n" + "=" * 70)
    print("COMPARING WITH DYNAMIC WORLD FILES")
    print("=" * 70)

    import glob

    # Find Dynamic World files
    dw_files = sorted(glob.glob(os.path.join(dynamic_world_dir, "*.nc")))
    if not dw_files:
        print("No Dynamic World files found!")
        return

    print(f"\nFound {len(dw_files)} Dynamic World files")

    # Load oldest and newest to see ID evolution
    oldest_file = dw_files[0]
    newest_file = dw_files[-1]

    print(f"\nOldest file: {os.path.basename(oldest_file)}")
    with xr.open_dataset(oldest_file, decode_times=False) as ds:
        old_ids = set(ds.id_geohash.values)
        print(f"  IDs in old file: {len(old_ids):,}")

    print(f"\nNewest file: {os.path.basename(newest_file)}")
    with xr.open_dataset(newest_file, decode_times=False) as ds:
        new_ids = set(ds.id_geohash.values)
        print(f"  IDs in new file: {len(new_ids):,}")

    # Vector IDs
    vector_ids = set(vector_gdf[id_column].values)
    print(f"\nVector file IDs: {len(vector_ids):,}")

    # Overlap analysis
    old_in_vector = len(old_ids & vector_ids)
    new_in_vector = len(new_ids & vector_ids)
    new_not_in_vector = len(new_ids - vector_ids)

    print(f"\n--- Overlap Analysis ---")
    print(f"Old file IDs in vector: {old_in_vector:,} ({old_in_vector / len(old_ids) * 100:.1f}%)")
    print(f"New file IDs in vector: {new_in_vector:,} ({new_in_vector / len(new_ids) * 100:.1f}%)")
    print(f"New IDs NOT in vector: {new_not_in_vector:,}")

    # This is key - these are the IDs you need to download but might not have geometries for!
    if new_not_in_vector > 0:
        print(f"\n⚠️ WARNING: {new_not_in_vector:,} lake IDs in Dynamic World are missing from vector file!")
        print("These IDs cannot be downloaded because there are no geometries.")

        # Show sample of missing IDs
        missing_ids = list(new_ids - vector_ids)[:10]
        print(f"Sample missing IDs: {missing_ids}")

    return old_ids, new_ids, vector_ids


def suggest_safe_chunk_sizes(gdf, id_column='id_geohash'):
    """
    Suggest safe chunk sizes based on actual memory profiling
    """
    print("\n" + "=" * 70)
    print("SUGGESTED CHUNK SIZES")
    print("=" * 70)

    ids = gdf[id_column].values
    unique_ids = np.unique(ids)

    # Test different chunk sizes
    test_sizes = [1000, 5000, 10000, 20000, 30000, 40000, 50000]

    print(f"\n{'Chunk Size':<12} {'Chunks':<10} {'Est. Memory (MB)':<20} {'Safe?'}")
    print("-" * 60)

    for size in test_sizes:
        if size > len(unique_ids):
            continue

        # Sample this many IDs
        sample_ids = unique_ids[:size]
        sample_gdf = gdf[gdf[id_column].isin(sample_ids)]
        memory_mb = sample_gdf.memory_usage(deep=True).sum() / 1024 / 1024

        n_chunks = len(unique_ids) // size + (1 if len(unique_ids) % size else 0)
        safe = "✓" if memory_mb < 8000 else "⚠️ HIGH"

        print(f"{size:<12,} {n_chunks:<10} {memory_mb:<20.1f} {safe}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Find optimal chunk size (under 4GB for safety)
    for size in test_sizes:
        if size > len(unique_ids):
            continue
        sample_ids = unique_ids[:size]
        sample_gdf = gdf[gdf[id_column].isin(sample_ids)]
        memory_mb = sample_gdf.memory_usage(deep=True).sum() / 1024 / 1024

        if memory_mb < 4000:  # Keep under 4GB for safety
            print(f"\n✅ Recommended chunk size: {size:,} lakes")
            print(f"   Memory usage: {memory_mb:.1f} MB")
            print(f"   Number of chunks: {len(unique_ids) // size + 1:,}")
            break


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()

    vector_file = os.environ['vector_lake_file']
    print(f"\nVector lake file: {vector_file}")
    dynamic_world_dir = os.environ['dynamic_world_dir']


    # Run analysis
    gdf = analyze_vector_file(vector_file)

    if gdf is not None:
        safe_chunk_size = test_split_strategies(gdf)
        compare_with_dynamic_world(dynamic_world_dir, gdf)
        suggest_safe_chunk_sizes(gdf)