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
    # ── Tropical Rainforest ──────────────────────────────────────────────────
    ("Amazon_Brazil",       [-62.0, -5.0,  -61.0, -4.0 ]),
    ("Congo_DRC",           [ 24.0, -2.0,   25.0, -1.0 ]),
    ("Borneo_Malaysia",     [114.0,  3.0,  115.0,  4.0 ]),

    # ── Desert / Arid ────────────────────────────────────────────────────────
    ("Sahara_Algeria",      [  3.0, 27.0,    4.0, 28.0 ]),
    ("Arabian_Saudi",       [ 45.0, 24.0,   46.0, 25.0 ]),
    ("Atacama_Chile",       [-69.0,-22.0,  -68.0,-21.0 ]),
    ("Gobi_Mongolia",       [105.0, 43.0,  106.0, 44.0 ]),

    # ── Temperate Forest ─────────────────────────────────────────────────────
    ("Pacific_NW_USA",      [-122.0, 47.0, -121.0, 48.0]),
    ("Central_Europe_DE",   [ 13.0,  51.0,  14.0,  52.0]),
    ("Patagonia_Argentina", [-71.0, -42.0, -70.0, -41.0]),

    # ── Snow / Arctic / Tundra ───────────────────────────────────────────────
    ("Siberia_Russia",      [ 90.0,  62.0,  91.0,  63.0]),
    ("Northern_Canada",     [-100.0, 60.0, -99.0,  61.0]),
    ("Greenland_Coast",     [-45.0,  67.0, -44.0,  68.0]),

    # ── Coastline / Ocean Interface ──────────────────────────────────────────
    ("Bay_of_Bengal_IN",    [ 80.0,  13.0,  81.0,  14.0]),
    ("Gulf_of_Mexico_USA",  [-90.0,  29.0, -89.0,  30.0]),
    ("Mediterranean_Italy", [ 12.0,  37.0,  13.0,  38.0]),

    # ── Urban / Built-up ─────────────────────────────────────────────────────
    ("Tokyo_Japan",         [139.0,  35.0, 140.0,  36.0]),
    ("London_UK",           [ -1.0,  51.0,   0.0,  52.0]),
    ("NYC_USA",             [-74.0,  40.0, -73.0,  41.0]),
    ("Mumbai_India",        [ 72.0,  19.0,  73.0,  20.0]),

    # ── Agricultural / Cropland ──────────────────────────────────────────────
    ("US_Midwest",          [-95.0,  41.0, -94.0,  42.0]),
    ("Ganges_Plain_India",  [ 78.0,  27.0,  79.0,  28.0]),
    ("Ukraine_Fields",      [ 32.0,  49.0,  33.0,  50.0]),

    # ── Savanna / Grassland ──────────────────────────────────────────────────
    ("East_Africa_Kenya",   [ 36.0,  -1.0,  37.0,   0.0]),
    ("Australian_Outback",  [133.0, -23.0, 134.0, -22.0]),
    ("South_Africa_Veld",   [ 26.0, -27.0,  27.0, -26.0]),

    # ── Mountain / High Altitude ─────────────────────────────────────────────
    ("Himalaya_Nepal",      [ 85.0,  28.0,  86.0,  29.0]),
    ("Andes_Peru",          [-75.0, -12.0, -74.0, -11.0]),
    ("Alps_Switzerland",    [  8.0,  46.0,   9.0,  47.0]),

    # ── Wetland / River Delta ────────────────────────────────────────────────
    ("Mekong_Delta",        [105.0,   9.0, 106.0,  10.0]),
    ("Nile_Delta_Egypt",    [ 31.0,  31.0,  32.0,  32.0]),
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
        num_scenes=100,
        scenes_per_region=4,   # Max 4 scenes per region = guaranteed geographic spread
        crop_size=512,
    )
