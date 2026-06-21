import os
import rasterio
from rasterio.windows import Window
import numpy as np

def fetch_planetary_computer_data(num_scenes=10, crop_size=512):
    """
    Automatically fetches cloud-free Thermal (B10) and RGB (B4,B3,B2) patches
    from Microsoft Planetary Computer's STAC API.
    """
    try:
        import pystac_client
        import planetary_computer
    except ImportError:
        print("Please install required packages first:")
        print("pip install pystac-client planetary-computer")
        return

    os.makedirs('data/train/ir', exist_ok=True)
    os.makedirs('data/train/rgb', exist_ok=True)
    
    print("Connecting to Microsoft Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    # Bounding box over a diverse area (e.g. Northern India)
    bbox = [76.0, 28.0, 78.0, 30.0]
    
    print(f"Searching for {num_scenes} cloud-free (<5%) Landsat 8/9 scenes...")
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        query={"eo:cloud_cover": {"lt": 5}}, 
        max_items=num_scenes
    )
    
    items = list(search.items())
    print(f"Found {len(items)} pristine scenes.")
    
    for i, item in enumerate(items):
        scene_id = item.id
        print(f"\n[{i+1}/{len(items)}] Downloading: {scene_id}")
        
        try:
            # We use rasterio windowed reading to ONLY download the 512x512 center patch
            # This makes the download take 2 seconds instead of downloading 500MB!
            with rasterio.open(item.assets["lwir11"].href) as src:
                w, h = src.width, src.height
                col_off, row_off = (w - crop_size) // 2, (h - crop_size) // 2
                window = Window(col_off, row_off, crop_size, crop_size)
                
                print("  -> Fetching Thermal (B10)...")
                ir_data = src.read(1, window=window)
                profile = src.profile
                profile.update(width=crop_size, height=crop_size, transform=src.window_transform(window))
                
            bands = []
            for b in ["red", "green", "blue"]:
                print(f"  -> Fetching Visible ({b.upper()})...")
                with rasterio.open(item.assets[b].href) as src:
                    bands.append(src.read(1, window=window))
            
            rgb_data = np.stack(bands)
            
            # Save files
            with rasterio.open(f'data/train/ir/{scene_id}.tif', 'w', **profile) as dst:
                dst.write(ir_data, 1)
                
            rgb_profile = profile.copy()
            rgb_profile.update(count=3)
            with rasterio.open(f'data/train/rgb/{scene_id}.tif', 'w', **rgb_profile) as dst:
                dst.write(rgb_data)
                
            print(f"  -> ✓ Saved to data/train/")
            
        except Exception as e:
            print(f"  -> ✗ Error processing {scene_id}: {e}")

if __name__ == '__main__':
    fetch_planetary_computer_data(num_scenes=20, crop_size=512)
