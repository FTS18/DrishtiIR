import numpy as np
import cv2
import os
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def generate_fractal_noise(shape, scale=1.5):
    """Generates 1/f fractal noise using Fast Fourier Transform (FFT)."""
    H, W = shape
    # 1. White noise
    noise = np.random.randn(H, W)
    
    # 2. To Frequency domain
    f = np.fft.fft2(noise)
    fshift = np.fft.fftshift(f)
    
    # 3. Create 1/f decay mask
    y, x = np.indices((H, W))
    center = (H // 2, W // 2)
    radius = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    radius[center[0], center[1]] = 1  # prevent divide by zero at DC component
    
    filter_mask = 1 / (radius ** scale)
    
    # 4. Apply mask and inverse FFT
    fshift_filtered = fshift * filter_mask
    f_filtered = np.fft.ifftshift(fshift_filtered)
    img = np.real(np.fft.ifft2(f_filtered))
    
    # 5. Normalize 0 to 1
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def generate_tile(idx, out_dir_ir, out_dir_rgb, size=256):
    """Generates a single synthetic biome tile and saves it to disk."""
    # Reset numpy seed for this process so all workers don't generate the same image
    np.random.seed()
    
    # ── 1. Terrain & Clouds ──
    terrain = generate_fractal_noise((size, size), scale=1.6)
    
    # Sparse clouds
    clouds = generate_fractal_noise((size, size), scale=1.8)
    clouds = np.clip((clouds - 0.6) * 2.5, 0, 1) 
    
    # ── 2. RGB Generation ──
    rgb = np.zeros((size, size, 3), dtype=np.float32)
    
    water_mask = terrain < 0.35
    veg_mask   = (terrain >= 0.35) & (terrain < 0.65)
    urban_mask = terrain >= 0.65
    
    rgb[water_mask] = [0.1, 0.25, 0.4]  # Deep Blue
    rgb[veg_mask]   = [0.15, 0.5, 0.2]  # Forest Green
    rgb[urban_mask] = [0.55, 0.5, 0.45] # Concrete/Sand
    
    # Add high-frequency texture
    texture = generate_fractal_noise((size, size), scale=0.8) * 0.15
    rgb += texture[..., np.newaxis]
    rgb = np.clip(rgb, 0, 1)
    
    # ── 3. Multi-Band IR Generation ──
    ir = np.zeros((size, size, 3), dtype=np.float32)
    
    # IR rules: Water absorbs heat (dark), Veg reflects NIR (bright), Urban emits thermal (bright)
    ir[water_mask] = [0.05, 0.05, 0.1]  # Dark in IR
    ir[veg_mask]   = [0.8, 0.6, 0.4]    # High NIR reflectance
    ir[urban_mask] = [0.6, 0.8, 0.9]    # High thermal emission
    
    ir += texture[..., np.newaxis]
    ir = np.clip(ir, 0, 1)
    
    # ── 4. Apply Clouds ──
    # RGB clouds are white/fluffy
    rgb = rgb * (1 - clouds[..., np.newaxis]) + clouds[..., np.newaxis] * np.array([0.9, 0.9, 0.9])
    
    # IR clouds are cold/opaque (darker gray)
    ir = ir * (1 - clouds[..., np.newaxis]) + clouds[..., np.newaxis] * np.array([0.3, 0.3, 0.3])
    
    # ── 5. Save to Disk ──
    rgb_8 = (rgb * 255).astype(np.uint8)
    ir_8  = (ir * 255).astype(np.uint8)
    
    # OpenCV expects BGR
    cv2.imwrite(os.path.join(out_dir_rgb, f"tile_{idx:05d}.jpg"), cv2.cvtColor(rgb_8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir_ir, f"tile_{idx:05d}.jpg"),  cv2.cvtColor(ir_8,  cv2.COLOR_RGB2BGR))

def worker(args):
    generate_tile(*args)

def main():
    parser = argparse.ArgumentParser(description="Massive Remote Sensing Dataset Generator")
    parser.add_argument("--count", type=int, default=300, help="Number of image pairs to generate")
    parser.add_argument("--size",  type=int, default=256,  help="Image resolution")
    args = parser.parse_args()
    
    out_ir = "data/train_massive/ir_multiband"
    out_rgb = "data/train_massive/rgb"
    os.makedirs(out_ir, exist_ok=True)
    os.makedirs(out_rgb, exist_ok=True)
    
    tasks = [(i, out_ir, out_rgb, args.size) for i in range(args.count)]
    
    print(f"\n{'='*60}")
    print(f"  Generating {args.count} High-Fidelity Fractal Synthetic Pairs")
    print(f"  Using {cpu_count()} CPU cores. This will be incredibly fast.")
    print(f"{'='*60}\n")
    
    # Multiprocessing pool
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap_unordered(worker, tasks), total=len(tasks)))
        
    print(f"\n[DONE] Saved {args.count} image pairs to data/train_massive/\n")

if __name__ == "__main__":
    main()
