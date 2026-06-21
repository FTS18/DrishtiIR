"""
stack_rgb.py
------------
Utility script to stack individual Landsat Red (B4), Green (B3), and Blue (B2) 
GeoTIFFs into a single 3-band RGB GeoTIFF required for DrishtiIR training.
"""

import os
import argparse
import rasterio

def stack_bands(b4_path: str, b3_path: str, b2_path: str, out_path: str):
    print(f"Reading Band 4 (Red): {b4_path}")
    print(f"Reading Band 3 (Green): {b3_path}")
    print(f"Reading Band 2 (Blue): {b2_path}")
    
    with rasterio.open(b4_path) as src4, \
         rasterio.open(b3_path) as src3, \
         rasterio.open(b2_path) as src2:
        
        # Ensure all bands have the same shape
        assert src4.shape == src3.shape == src2.shape, "All bands must have the same dimensions."
        
        # Read the data
        red = src4.read(1)
        green = src3.read(1)
        blue = src2.read(1)
        
        # Copy metadata from one of the bands
        meta = src4.meta.copy()
        
        # Update metadata for 3 bands
        meta.update({
            "count": 3,
            "dtype": red.dtype
        })
        
        print(f"Writing stacked RGB to: {out_path}...")
        with rasterio.open(out_path, 'w', **meta) as dest:
            dest.write(red, 1)
            dest.write(green, 2)
            dest.write(blue, 3)
            
    print("Done! 🎉")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stack Landsat B4, B3, B2 into an RGB GeoTIFF.")
    parser.add_argument("--b4", required=True, help="Path to Band 4 (Red) .tif")
    parser.add_argument("--b3", required=True, help="Path to Band 3 (Green) .tif")
    parser.add_argument("--b2", required=True, help="Path to Band 2 (Blue) .tif")
    parser.add_argument("--out", required=True, help="Path to output RGB .tif")
    
    args = parser.parse_args()
    stack_bands(args.b4, args.b3, args.b2, args.out)
