"""
fetch_massive_dataset.py
------------------------
Downloads paired Landsat 8/9 IR + RGB scenes from Microsoft Planetary Computer.

Samples from 20 globally diverse geographic regions covering every major
biome and terrain type so the model generalizes to the whole world:
  - Tropical rainforest (Amazon, Congo)
  - Desert / arid (Sahara, Arabian, Atacama)
  - Temperate forest (Pacific NW, Central Europe)
  - Snow / tundra (Siberia, Canada)
  - Coastline / ocean interface (Bay of Bengal, Gulf of Mexico)
  - Urban (London, Tokyo, NYC, Mumbai)
  - Agricultural (US Midwest, Ganges Plain)
  - Savanna / grassland (East Africa, Australian outback)
"""

import os
import rasterio
import numpy as np
import time
import random

try:
    import pystac_client
    import planetary_computer
except ImportError:
    print("Please install: pip install pystac-client planetary-computer")
    import sys
    sys.exit(1)


# ─── Global Region Definitions ────────────────────────────────────────────────
# Each entry: (name, [lon_min, lat_min, lon_max, lat_max])
# Spread across all continents, biomes, and climate zones.

GLOBAL_REGIONS = [
    # ── India (ISRO Focus) - Diverse Terrains ────────────────────────────────
    ("Mumbai_Urban",        [ 72.5,  18.5,  73.5,  19.5]),
    ("Delhi_NCR_Urban",     [ 76.5,  28.0,  77.5,  29.0]),
    ("Bangalore_Urban",     [ 77.0,  12.5,  78.0,  13.5]),
    ("Kolkata_Wetland",     [ 88.0,  22.0,  89.0,  23.0]),
    ("Chennai_Coast",       [ 80.0,  12.5,  81.0,  13.5]),
    ("Hyderabad_Deccan",    [ 78.0,  17.0,  79.0,  18.0]),
    ("Ganges_Agri_UP",      [ 79.0,  26.5,  80.0,  27.5]),
    ("Punjab_Farms",        [ 75.0,  30.0,  76.0,  31.0]),
    ("Kerala_Backwaters",   [ 76.0,   9.0,  77.0,  10.0]),
    ("Western_Ghats",       [ 74.0,  15.0,  75.0,  16.0]),
    ("Assam_Brahmaputra",   [ 92.0,  26.0,  93.0,  27.0]),

    # ── High-Value Global Urban ──────────────────────────────────────────────
    ("Tokyo_Japan",         [139.0,  35.0, 140.0,  36.0]),
    ("London_UK",           [ -0.5,  51.0,   0.5,  51.5]),
    ("NYC_USA",             [-74.5,  40.5, -73.5,  41.5]),
    ("Paris_France",        [  2.0,  48.5,   2.5,  49.0]),
    ("Cairo_Egypt",         [ 31.0,  30.0,  31.5,  30.5]),

    # ── High-Value Global Agriculture & Forest ───────────────────────────────
    ("US_Midwest_Farms",    [-95.0,  41.0, -94.0,  42.0]),
    ("Ukraine_Fields",      [ 32.0,  49.0,  33.0,  50.0]),
    ("Amazon_Rainforest",   [-62.0,  -5.0, -61.0,  -4.0]),
    ("Central_Europe_DE",   [ 13.0,  51.0,  14.0,  52.0]),
    ("Pacific_NW_USA",      [-122.0, 47.0, -121.0, 48.0]),

    # ── High-Value River Deltas (Water/Land boundary) ────────────────────────
    ("Mekong_Delta",        [105.0,   9.0, 106.0,  10.0]),
    ("Nile_Delta",          [ 31.0,  31.0,  32.0,  32.0]),
]


def fetch_massive_dataset(
    num_scenes: int = 100,
    scenes_per_region: int = 4,
    crop_size: int = 512,
    output_dir: str = 'data/train_massive',
    cloud_threshold: float = 10.0,
):
    """
    Download paired IR + RGB Landsat scenes sampled across globally diverse regions.

    Args:
        num_scenes       : Total target number of scenes to download
        scenes_per_region: Max scenes to take from each region (prevents any
                           single region from dominating the dataset)
        crop_size        : Pixel size of the center crop saved per scene
        output_dir       : Output directory (ir_multiband/ and rgb/ subdirs)
        cloud_threshold  : Max cloud cover % to accept (default: 10%)
    """
    os.makedirs(f'{output_dir}/ir_multiband', exist_ok=True)
    os.makedirs(f'{output_dir}/rgb', exist_ok=True)

    print("Connecting to Microsoft Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Shuffle regions so we get variety even if we stop early
    regions = list(GLOBAL_REGIONS)
    random.shuffle(regions)

    all_candidates = []  # (region_name, stac_item)

    print(f"\nSearching {len(regions)} global regions for pristine scenes (cloud < {cloud_threshold}%)...\n")
    for region_name, bbox in regions:
        try:
            search = catalog.search(
                collections=["landsat-c2-l2"],
                bbox=bbox,
                datetime="2022-01-01/2024-12-31",  # 3-year window for more candidates
                max_items=30,                        # Small per-region fetch = fast
            )
            items = list(search.items())
            clean = [it for it in items if it.properties.get("eo:cloud_cover", 100) < cloud_threshold]
            clean = clean[:scenes_per_region]
            print(f"  [{region_name:25s}] {len(items):3d} raw → {len(clean)} clean scenes")
            for item in clean:
                all_candidates.append((region_name, item))
        except Exception as e:
            print(f"  [{region_name:25s}] SKIP: {e}")
        time.sleep(0.2)  # Be polite to the API

    # Shuffle so we don't get geographically clustered batches
    random.shuffle(all_candidates)
    selected = all_candidates[:num_scenes]

    print(f"\nSelected {len(selected)} scenes from {len(set(r for r,_ in selected))} regions")
    print(f"Downloading {len(selected)} scenes...\n")

    success_count = 0
    for i, (region_name, item) in enumerate(selected):
        scene_id = item.id
        print(f"[{i+1:03d}/{len(selected)}] {region_name} | {scene_id}")

        try:
            nir_url     = item.assets["nir08"].href
            swir1_url   = item.assets["swir16"].href
            swir2_url   = item.assets["swir22"].href
            thermal_url = item.assets["lwir11"].href
            red_url     = item.assets["red"].href
            green_url   = item.assets["green"].href
            blue_url    = item.assets["blue"].href
        except KeyError as e:
            print(f"  -> Missing band {e}, skipping...")
            continue

        try:
            with rasterio.Env(GDAL_HTTP_RETRY_DELAY=1):
                with rasterio.open(thermal_url) as src:
                    h, w   = src.height, src.width
                    cx, cy = w // 2, h // 2
                    half   = crop_size // 2
                    # Clamp window to valid bounds
                    x_off = max(0, min(cx - half, w - crop_size))
                    y_off = max(0, min(cy - half, h - crop_size))
                    window = rasterio.windows.Window(x_off, y_off, crop_size, crop_size)
                    thermal = src.read(1, window=window)

                with rasterio.open(nir_url)   as src: nir   = src.read(1, window=window)
                with rasterio.open(swir1_url) as src: swir1 = src.read(1, window=window)
                with rasterio.open(swir2_url) as src: swir2 = src.read(1, window=window)

                ir_stack = np.stack([nir, swir1, swir2, thermal], axis=0)
                np.save(f'{output_dir}/ir_multiband/{region_name}_{scene_id}.npy', ir_stack)

                with rasterio.open(red_url)   as src: red   = src.read(1, window=window)
                with rasterio.open(green_url) as src: green = src.read(1, window=window)
                with rasterio.open(blue_url)  as src: blue  = src.read(1, window=window)

                rgb_stack = np.stack([red, green, blue], axis=0)
                np.save(f'{output_dir}/rgb/{region_name}_{scene_id}.npy', rgb_stack)

            success_count += 1
            print(f"  -> Success! ({success_count} total)")
        except Exception as e:
            print(f"  -> Error: {e}")

        time.sleep(0.3)

    print(f"\n{'='*60}")
    print(f"  Download complete: {success_count}/{len(selected)} scenes")
    print(f"  Saved to: {output_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    fetch_massive_dataset(
        num_scenes=200,        # Increased total target
        scenes_per_region=10,  # Allow up to 10 from good regions
        crop_size=512,
    )
