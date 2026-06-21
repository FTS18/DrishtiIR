import os
import rasterio
import numpy as np
import time

try:
    import pystac_client
    import planetary_computer
except ImportError:
    print("Please install: pip install pystac-client planetary-computer")
    import sys
    sys.exit(1)

def fetch_massive_dataset(num_scenes=100, crop_size=512, output_dir='data/train_massive'):
    os.makedirs(f'{output_dir}/ir_multiband', exist_ok=True)
    os.makedirs(f'{output_dir}/rgb', exist_ok=True)
    
    print("Connecting to Microsoft Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    print("Fetching raw Landsat items (bypassing cloud-cover server timeout)...")
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=[77.0, 28.0, 78.0, 29.0],
        datetime="2023-01-01/2023-12-31",
        max_items=400
    )
    
    try:
        all_items = list(search.items())
    except Exception as e:
        print(f"Failed to fetch from Microsoft: {e}")
        return
        
    print(f"Fetched {len(all_items)} raw scenes instantly.")
    
    pristine_items = [item for item in all_items if item.properties.get("eo:cloud_cover", 100) < 5]
    items = pristine_items[:num_scenes]
    print(f"Found {len(items)} pristine low-cloud scenes after local filtering.")
    
    success_count = 0
    for i, item in enumerate(items):
        scene_id = item.id
        print(f"\n[{i+1}/{len(items)}] Downloading: {scene_id}")
        
        try:
            nir_url = item.assets["nir08"].href
            swir1_url = item.assets["swir16"].href
            swir2_url = item.assets["swir22"].href
            thermal_url = item.assets["lwir11"].href
            
            red_url = item.assets["red"].href
            green_url = item.assets["green"].href
            blue_url = item.assets["blue"].href
        except KeyError:
            print(f"  -> Missing required bands in {scene_id}, skipping...")
            continue
            
        try:
            with rasterio.Env(GDAL_HTTP_RETRY_DELAY=1):
                with rasterio.open(thermal_url) as src:
                    h, w = src.height, src.width
                    cx, cy = w // 2, h // 2
                    half = crop_size // 2
                    window = rasterio.windows.Window(cx - half, cy - half, crop_size, crop_size)
                    thermal = src.read(1, window=window)
                with rasterio.open(nir_url) as src:
                    nir = src.read(1, window=window)
                with rasterio.open(swir1_url) as src:
                    swir1 = src.read(1, window=window)
                with rasterio.open(swir2_url) as src:
                    swir2 = src.read(1, window=window)
                    
                ir_stack = np.stack([nir, swir1, swir2, thermal], axis=0)
                np.save(f'{output_dir}/ir_multiband/{scene_id}.npy', ir_stack)
                
                with rasterio.open(red_url) as src:
                    red = src.read(1, window=window)
                with rasterio.open(green_url) as src:
                    green = src.read(1, window=window)
                with rasterio.open(blue_url) as src:
                    blue = src.read(1, window=window)
                    
                rgb_stack = np.stack([red, green, blue], axis=0)
                np.save(f'{output_dir}/rgb/{scene_id}.npy', rgb_stack)
                
            success_count += 1
            print(f"  -> Success!")
        except Exception as e:
            print(f"  -> Error processing {scene_id}: {e}")
            
        time.sleep(0.5)
        
    print(f"\nFinished! Successfully downloaded {success_count} multi-band pairs.")

if __name__ == '__main__':
    fetch_massive_dataset(num_scenes=100, crop_size=512)
