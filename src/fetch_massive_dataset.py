import os
import rasterio
from rasterio.windows import Window
import numpy as np
import time

def fetch_massive_dataset(num_scenes=1000, crop_size=256, output_dir='data/train_massive'):
    """
    Massive fetcher designed for Kaggle.
    Downloads 3-channel input: B10 (Thermal), B6 (SWIR), B5 (NIR).
    Downloads 3-channel target: B4 (Red), B3 (Green), B2 (Blue).
    """
    try:
        import pystac_client
        import planetary_computer
    except ImportError:
        print("Please install: pip install pystac-client planetary-computer")
        return

    os.makedirs(f'{output_dir}/ir_multiband', exist_ok=True)
    os.makedirs(f'{output_dir}/rgb', exist_ok=True)
    
    print("Connecting to Microsoft Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    # Smaller bounding box (Central India region) to prevent API timeouts
    bbox = [77.0, 20.0, 79.0, 22.0]
    
    print(f"Searching for up to {num_scenes} low-cloud (<5%) Landsat 8/9 scenes...")
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime="2023-01-01/2023-12-31",
        query={"eo:cloud_cover": {"lt": 5}}, 
        max_items=num_scenes
    )
    
    items = list(search.items())
    print(f"Found {len(items)} pristine scenes.")
    
    success_count = 0
    for i, item in enumerate(items):
        scene_id = item.id
        print(f"\n[{i+1}/{len(items)}] Downloading: {scene_id}")
        
        try:
            # We use rasterio windowed reading to ONLY download the center patch
            with rasterio.open(item.assets["lwir11"].href) as src:
                w, h = src.width, src.height
                col_off, row_off = (w - crop_size) // 2, (h - crop_size) // 2
                window = Window(col_off, row_off, crop_size, crop_size)
                
                print("  -> Fetching Thermal (B10)...")
                b10_data = src.read(1, window=window)
                profile = src.profile
                profile.update(width=crop_size, height=crop_size, transform=src.window_transform(window), count=3)
                
            print("  -> Fetching SWIR (B6)...")
            with rasterio.open(item.assets["swir16"].href) as src:
                b6_data = src.read(1, window=window)
                
            print("  -> Fetching NIR (B5)...")
            with rasterio.open(item.assets["nir08"].href) as src:
                b5_data = src.read(1, window=window)
                
            input_multiband = np.stack([b10_data, b6_data, b5_data])
            
            bands_rgb = []
            for b in ["red", "green", "blue"]:
                print(f"  -> Fetching Visible ({b.upper()})...")
                with rasterio.open(item.assets[b].href) as src:
                    bands_rgb.append(src.read(1, window=window))
            
            rgb_data = np.stack(bands_rgb)
            
            # Save files
            with rasterio.open(f'{output_dir}/ir_multiband/{scene_id}.tif', 'w', **profile) as dst:
                dst.write(input_multiband)
                
            with rasterio.open(f'{output_dir}/rgb/{scene_id}.tif', 'w', **profile) as dst:
                dst.write(rgb_data)
                
            success_count += 1
            print(f"  -> ✓ Saved to {output_dir}")
            
        except Exception as e:
            print(f"  -> ✗ Error processing {scene_id}: {e}")
            
        time.sleep(0.5) # Avoid hammering the API
        
    print(f"\nFinished! Successfully downloaded {success_count} multi-band pairs.")

if __name__ == '__main__':
    # Scaling to 500 for massive Kaggle run
    fetch_massive_dataset(num_scenes=500, crop_size=512)
