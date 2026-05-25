import xarray as xr
import os
import glob
import numpy as np
from dotenv import load_dotenv

load_dotenv()

dynamic_world_dir = os.environ.get('dynamic_world_dir')
split_new_dynamic_world_data_dir = os.environ.get('split_new_dynamic_world_data_dir')

print("=" * 80)
print("CHECKING LAKE ID OVERLAP")
print("=" * 80)

# Load existing file
existing_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
existing_file = max(existing_files, key=os.path.getctime)
print(f"\nExisting file: {os.path.basename(existing_file)}")

with xr.open_dataset(existing_file, decode_times=False) as ds:
    existing_ids = set(ds.id_geohash.values)
    print(f"  Total lakes: {len(existing_ids):,}")
    print(f"  Sample IDs (first 20): {list(existing_ids)[:20]}")
    print(f"  ID prefix distribution:")
    prefixes = {}
    for lake_id in list(existing_ids)[:1000]:
        prefix = lake_id[:3]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    print(f"    {prefixes}")

# Check a few chunk files
chunk_files = sorted(glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc")))
print(f"\nChunk files: {len(chunk_files)} total")

all_chunk_ids = set()
sample_chunks = chunk_files[:5]  # Check first 5 chunks

for chunk_file in sample_chunks:
    with xr.open_dataset(chunk_file, decode_times=False) as ds:
        chunk_ids = set(ds.id_geohash.values)
        all_chunk_ids.update(chunk_ids)
        print(f"\n  {os.path.basename(chunk_file)}:")
        print(f"    Lakes: {len(chunk_ids):,}")
        print(f"    Sample IDs (first 10): {list(chunk_ids)[:10]}")
        print(f"    ID prefix distribution (first 100):")
        prefixes = {}
        for lake_id in list(chunk_ids)[:100]:
            prefix = lake_id[:3]
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        print(f"      {prefixes}")

# Check overlap
print(f"\n" + "=" * 80)
print("OVERLAP ANALYSIS")
print("=" * 80)

print(f"\nTotal unique chunk IDs sampled: {len(all_chunk_ids):,}")
print(f"Total existing IDs: {len(existing_ids):,}")

# Take a sample of chunk IDs for overlap check (to avoid memory issues)
chunk_sample = list(all_chunk_ids)[:10000]
overlap = existing_ids & set(chunk_sample)

print(f"\nOverlap in sample (first 10,000 chunk IDs): {len(overlap)}")
if overlap:
    print(f"  Example overlapping IDs: {list(overlap)[:10]}")
else:
    print(f"  NO OVERLAP FOUND!")

    # Check if prefixes just differ
    existing_prefixes = set([lake_id[:3] for lake_id in existing_ids])
    chunk_prefixes = set([lake_id[:3] for lake_id in all_chunk_ids])
    print(f"\n  Existing ID prefixes: {existing_prefixes}")
    print(f"  Chunk ID prefixes: {chunk_prefixes}")

    if existing_prefixes & chunk_prefixes:
        print(f"  Shared prefixes: {existing_prefixes & chunk_prefixes}")
    else:
        print(f"  NO SHARED PREFIXES - Different lake ID systems!")

print("\n" + "=" * 80)