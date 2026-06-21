import os
import time
import requests
import tarfile
import rasterio
import numpy as np
import shutil

USERNAME = "ananay"
APP_TOKEN = "Mr7bzqayU4n7AaNFDaMRsJY0lKpnQ53Aap523URz7Me7xVZKrX16YbT7w8L6utZF"
M2M_API = "https://m2m.cr.usgs.gov/api/api/json/stable"

def send_request(endpoint, data=None, api_key=None):
    if data is None: data = {}
    headers = {}
    if api_key: headers['X-Auth-Token'] = api_key
    resp = requests.post(f"{M2M_API}/{endpoint}", json=data, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"API Error: {resp.text}")
    res = resp.json()
    if res.get('errorCode'):
        raise Exception(f"M2M Error {res.get('errorCode')}: {res.get('errorMessage')}")
    return res['data']

def fetch_massive_dataset(num_scenes=50, crop_size=512, output_dir='data/train_massive'):
    os.makedirs(f'{output_dir}/ir_multiband', exist_ok=True)
    os.makedirs(f'{output_dir}/rgb', exist_ok=True)
    tmp_dir = f'{output_dir}/tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    
    print("Logging into USGS M2M API...")
    try:
        api_key = send_request("login-token", {"username": USERNAME, "token": APP_TOKEN})
        print("Authenticated successfully.")
    except Exception as e:
        print(f"Login failed: {e}")
        return
    
    dataset_name = "landsat_ot_c2_l2"
    
    print("Searching for Landsat 8/9 Collection 2 Level 2 scenes...")
    search_payload = {
        "datasetName": dataset_name,
        "maxResults": num_scenes,
        "sceneFilter": {
            "spatialFilter": {
                "filterType": "mbr",
                "lowerLeft": {"latitude": 28.0, "longitude": 77.0},
                "upperRight": {"latitude": 29.0, "longitude": 78.0}
            },
            "cloudCoverFilter": {"max": 5, "min": 0, "includeUnknown": False}
        }
    }
    scenes_data = send_request("scene-search", search_payload, api_key)
    results = scenes_data.get('results', [])
    print(f"Found {len(results)} cloud-free scenes in the Delhi region.")
    
    entity_ids = [r['entityId'] for r in results]
    if not entity_ids: return
    
    # Get download options
    print("Checking availability on USGS servers...")
    options_payload = {
        "datasetName": dataset_name,
        "entityIds": entity_ids
    }
    options_data = send_request("download-options", options_payload, api_key)
    
    downloads = []
    for opt in options_data:
        # We need the full product bundle for C2 L2
        if opt['available'] and 'Bundle' in opt['productName']:
            downloads.append({"entityId": opt['entityId'], "productId": opt['id']})
    
    if not downloads:
        print("No immediate downloads available without ordering.")
        return
        
    print(f"Requesting {len(downloads)} direct download URLs...")
    dl_request = {
        "downloads": downloads,
        "label": "hackathon_dl"
    }
    dl_data = send_request("download-request", dl_request, api_key)
    
    urls = [dl['url'] for dl in dl_data['availableDownloads']]
    
    print("Starting processing pipeline...")
    success_count = 0
    
    for i, (entity_id, url) in enumerate(zip([d['entityId'] for d in downloads], urls)):
        print(f"\n[{i+1}/{len(downloads)}] Processing {entity_id}...")
        tar_path = os.path.join(tmp_dir, f"{entity_id}.tar")
        
        # 1. Download
        print(f"  -> Downloading ~1GB tar bundle from USGS...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(tar_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192*16): 
                        f.write(chunk)
        except Exception as e:
            print(f"  -> Error downloading: {e}")
            continue
                    
        # 2. Extract
        print(f"  -> Extracting necessary TIF bands...")
        band_paths = {}
        try:
            with tarfile.open(tar_path) as tar:
                for member in tar.getmembers():
                    if "B2.TIF" in member.name: band_paths['blue'] = member.name
                    elif "B3.TIF" in member.name: band_paths['green'] = member.name
                    elif "B4.TIF" in member.name: band_paths['red'] = member.name
                    elif "B5.TIF" in member.name: band_paths['nir'] = member.name
                    elif "B6.TIF" in member.name: band_paths['swir1'] = member.name
                    elif "B7.TIF" in member.name: band_paths['swir2'] = member.name
                    elif "B10.TIF" in member.name: band_paths['thermal'] = member.name
                    else: continue
                    tar.extract(member, tmp_dir)
        except Exception as e:
            print(f"  -> Error extracting tar: {e}")
            if os.path.exists(tar_path): os.remove(tar_path)
            continue
                
        if len(band_paths) < 7:
            print(f"  -> Missing required bands in bundle, skipping.")
            if os.path.exists(tar_path): os.remove(tar_path)
            continue
            
        # 3. Process into Numpy Arrays
        print(f"  -> Cropping and converting to NumPy arrays...")
        try:
            arrays = {}
            for k, p in band_paths.items():
                full_path = os.path.join(tmp_dir, p)
                with rasterio.open(full_path) as src:
                    h, w = src.height, src.width
                    cx, cy = w // 2, h // 2
                    half = crop_size // 2
                    window = rasterio.windows.Window(cx - half, cy - half, crop_size, crop_size)
                    arrays[k] = src.read(1, window=window)
                os.remove(full_path) # Cleanup TIF immediately to save disk space
                
            ir_stack = np.stack([arrays['nir'], arrays['swir1'], arrays['swir2'], arrays['thermal']], axis=0)
            rgb_stack = np.stack([arrays['red'], arrays['green'], arrays['blue']], axis=0)
            
            np.save(f'{output_dir}/ir_multiband/{entity_id}.npy', ir_stack)
            np.save(f'{output_dir}/rgb/{entity_id}.npy', rgb_stack)
            success_count += 1
            print(f"  -> Success! Cleaned up disk space.")
        except Exception as e:
            print(f"  -> Processing error: {e}")
            
        # 4. Cleanup Tar
        if os.path.exists(tar_path): 
            os.remove(tar_path)
            
    # Final cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    print("Logging out of USGS M2M API...")
    send_request("logout", {}, api_key)
    
    print(f"\nFinished! Successfully downloaded {success_count} multi-band pairs.")

if __name__ == '__main__':
    fetch_massive_dataset(num_scenes=50, crop_size=512)
