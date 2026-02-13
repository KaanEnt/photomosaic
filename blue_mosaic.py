"""
blue_mosaic.py - Standalone "Blue In, Blue Out" mosaic generator

Self-contained mosaic script that does NOT depend on faiss, dlib, or the
emosaic package. Uses only Pillow, OpenCV, and NumPy — all of which are
already in Venv_001.

The algorithm is the same as worldveil/photomosaic:
  1. Load tile images from a codebook directory
  2. Resize each tile to the target cell size and flatten to a vector
  3. Divide the target image into a grid of cells
  4. For each cell, find the tile with the smallest L2 distance
  5. Place the best-matching tile into the output mosaic

Usage:
    conda activate Venv_001

    # Full pipeline: generate tiles, convert target, build mosaic
    python blue_mosaic.py \
        --target tests/space.png \
        --output ./output/space_mosaic.jpg \
        --tile-mode blocks \
        --tile-count 50 \
        --color "#0000FF" \
        --scale 12

    # Use existing tile folder (e.g. from prepare_tiles.py)
    python blue_mosaic.py \
        --target tests/space.png \
        --output ./output/space_mosaic.jpg \
        --codebook-dir ./my_blue_tiles \
        --scale 12

    # Use real photos (converts them to blue first)
    python blue_mosaic.py \
        --target tests/space.png \
        --output ./output/space_mosaic.jpg \
        --source-images ./family_photos \
        --color "#0000FF" \
        --scale 10
"""

import os
import sys
import glob
import time
import argparse
import random

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps


# ─── Color utilities ─────────────────────────────────────────────────────────

def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


# ─── Tile generation (built-in, no separate script needed) ──────────────────

def generate_tiles_blocks(color_rgb, black_rgb, count, tile_w, tile_h):
    """Pixel-block grid tiles with varying fill density."""
    tiles = []
    cells_x = max(3, tile_w // 6)
    cells_y = max(3, tile_h // 6)
    cell_w = tile_w / cells_x
    cell_h = tile_h / cells_y
    total_cells = cells_x * cells_y

    for i in range(count):
        t = i / max(count - 1, 1)
        img = Image.new('RGB', (tile_w, tile_h), black_rgb)
        draw = ImageDraw.Draw(img)

        n_filled = int(total_cells * t)
        rng = random.Random(42 + i)
        all_cells = [(cx, cy) for cy in range(cells_y) for cx in range(cells_x)]
        rng.shuffle(all_cells)
        filled = set(tuple(c) for c in all_cells[:n_filled])
        fill = lerp_color(black_rgb, color_rgb, 0.4 + 0.6 * t)

        for cy in range(cells_y):
            for cx in range(cells_x):
                if (cx, cy) in filled:
                    x0 = int(cx * cell_w)
                    y0 = int(cy * cell_h)
                    x1 = int((cx + 1) * cell_w) - 1
                    y1 = int((cy + 1) * cell_h) - 1
                    draw.rectangle([x0, y0, x1, y1], fill=fill)

        # Convert PIL -> OpenCV (BGR)
        arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
        tiles.append(arr)
    return tiles


def generate_tiles_halftone(color_rgb, black_rgb, count, tile_w, tile_h):
    """Dot-grid halftone tiles with varying dot radius."""
    tiles = []
    dots_x = max(2, tile_w // 12)
    dots_y = max(2, tile_h // 12)
    spacing_x = tile_w / dots_x
    spacing_y = tile_h / dots_y
    max_radius = min(spacing_x, spacing_y) * 0.48

    for i in range(count):
        t = i / max(count - 1, 1)
        img = Image.new('RGB', (tile_w, tile_h), black_rgb)
        draw = ImageDraw.Draw(img)

        if t > 0.01:
            radius = max(1, max_radius * t)
            fill = lerp_color(black_rgb, color_rgb, 0.5 + 0.5 * t)
            for dy in range(dots_y):
                for dx in range(dots_x):
                    cx = spacing_x * (dx + 0.5)
                    cy = spacing_y * (dy + 0.5)
                    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=fill)

        arr = np.array(img)[:, :, ::-1]
        tiles.append(arr)
    return tiles


def generate_tiles_crosshatch(color_rgb, black_rgb, count, tile_w, tile_h):
    """Line-hatching tiles with increasing density."""
    import math
    tiles = []
    for i in range(count):
        t = i / max(count - 1, 1)
        img = Image.new('RGB', (tile_w, tile_h), black_rgb)
        draw = ImageDraw.Draw(img)

        if t > 0.02 and t <= 0.95:
            fill = lerp_color(black_rgb, color_rgb, 0.5 + 0.5 * t)
            lw = max(1, int(1 + t * 2))
            spacing = max(3, int(20 * (1 - t) + 3))
            diag = int(math.sqrt(tile_w**2 + tile_h**2))

            for off in range(-diag, diag, spacing):
                draw.line([(off, 0), (off + tile_h, tile_h)], fill=fill, width=lw)
            if t > 0.3:
                for off in range(-diag, diag, spacing):
                    draw.line([(tile_w - off, 0), (tile_w - off - tile_h, tile_h)], fill=fill, width=lw)
            if t > 0.6:
                for y in range(0, tile_h, spacing):
                    draw.line([(0, y), (tile_w, y)], fill=fill, width=lw)
            if t > 0.8:
                for x in range(0, tile_w, spacing):
                    draw.line([(x, 0), (x, tile_h)], fill=fill, width=lw)
        elif t > 0.95:
            img = Image.new('RGB', (tile_w, tile_h), color_rgb)

        arr = np.array(img)[:, :, ::-1]
        tiles.append(arr)
    return tiles


def generate_tiles_noise(color_rgb, black_rgb, count, tile_w, tile_h):
    """Dithered noise tiles with varying pixel density."""
    tiles = []
    for i in range(count):
        t = i / max(count - 1, 1)
        rng = np.random.RandomState(seed=42 + i)
        mask = rng.random((tile_h, tile_w)) < t
        fill = lerp_color(black_rgb, color_rgb, 0.5 + 0.5 * t)

        arr = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        arr[mask] = fill
        arr[~mask] = black_rgb
        # This array is already in RGB order; convert to BGR for OpenCV
        tiles.append(arr[:, :, ::-1].copy())
    return tiles


TILE_GENERATORS = {
    "blocks": generate_tiles_blocks,
    "halftone": generate_tiles_halftone,
    "crosshatch": generate_tiles_crosshatch,
    "noise": generate_tiles_noise,
}


# ─── Load tiles from disk ───────────────────────────────────────────────────

def load_tiles_from_dir(codebook_dir, tile_h, tile_w):
    """Load images from a directory, resize to tile dimensions."""
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(codebook_dir, pat)))

    if not paths:
        print(f"ERROR: No images found in {codebook_dir}")
        sys.exit(1)

    tiles = []
    for p in sorted(paths):
        img = cv2.imread(p)
        if img is None:
            continue
        resized = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        tiles.append(resized)

    print(f"  Loaded {len(tiles)} tiles from {codebook_dir}")
    return tiles


# ─── Convert source photos to monochrome ────────────────────────────────────

def convert_photos_to_blue(source_dir, output_dir, color_hex, black_hex, tile_h, tile_w):
    """Convert real photos to monochrome, resize to tile dims, return as list."""
    os.makedirs(output_dir, exist_ok=True)
    extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

    files = [f for f in os.listdir(source_dir) if f.lower().endswith(extensions)]
    if not files:
        print(f"ERROR: No images found in {source_dir}")
        sys.exit(1)

    tiles = []
    for filename in sorted(files):
        try:
            with Image.open(os.path.join(source_dir, filename)) as img:
                if img.mode in ('RGBA', 'LA', 'PA'):
                    img = img.convert('RGB')
                gray = ImageOps.grayscale(img)
                mono = ImageOps.colorize(gray, black=black_hex, white=color_hex)
                mono = mono.convert('RGB')

                # Save to disk (for inspection) and also keep in memory
                out_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.jpg')
                mono.save(out_path, 'JPEG', quality=95)

                # Convert to OpenCV BGR array at tile size
                arr = np.array(mono)[:, :, ::-1]  # RGB -> BGR
                resized = cv2.resize(arr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                tiles.append(resized)
        except Exception as e:
            print(f"  [SKIP] {filename}: {e}")

    print(f"  Converted {len(tiles)} photos to monochrome tiles")
    return tiles


# ─── Core mosaic algorithm ──────────────────────────────────────────────────

def build_tile_index(tiles, tile_h, tile_w):
    """
    Flatten each tile into a vector and stack into a matrix.
    Returns (matrix of shape [N, D], list of tile arrays).
    """
    vectors = []
    for tile in tiles:
        # Resize to exact tile dims (should already be, but ensure)
        if tile.shape[0] != tile_h or tile.shape[1] != tile_w:
            tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        v = tile.reshape(1, -1).astype(np.float32)
        vectors.append(v)

    matrix = np.vstack(vectors)  # shape: (N, tile_h * tile_w * 3)
    return matrix


def find_nearest_tile(query_vector, tile_matrix):
    """Brute-force L2 nearest neighbor. Returns (index, distance)."""
    # query_vector: (1, D), tile_matrix: (N, D)
    diffs = tile_matrix - query_vector  # (N, D)
    dists = np.sum(diffs ** 2, axis=1)  # (N,)
    idx = np.argmin(dists)
    return idx, dists[idx]


def find_nearest_tile_topk(query_vector, tile_matrix, k):
    """Return k nearest tile indices, pick one randomly."""
    diffs = tile_matrix - query_vector
    dists = np.sum(diffs ** 2, axis=1)
    top_k_idx = np.argpartition(dists, k)[:k]
    return random.choice(top_k_idx)


def mosaicify(target_image, tile_h, tile_w, tile_matrix, tiles, best_k=1, opacity=0.0):
    """
    Build a mosaic from the target image using the tile codebook.
    Same algorithm as emosaic.mosaicify but with pure NumPy L2 search.
    """
    img_h, img_w, channels = target_image.shape

    # Grid layout
    n_rows = img_h // tile_h
    n_cols = img_w // tile_w
    h_offset = (img_h % tile_h) // 2
    w_offset = (img_w % tile_w) // 2

    mosaic = np.zeros_like(target_image)
    total = n_rows * n_cols
    count = 0

    print(f"  Grid: {n_cols}x{n_rows} = {total} tiles")

    for row in range(n_rows):
        for col in range(n_cols):
            x = row * tile_h + h_offset
            y = col * tile_w + w_offset

            # Extract target patch and vectorize
            patch = target_image[x:x + tile_h, y:y + tile_w]
            if patch.shape[0] != tile_h or patch.shape[1] != tile_w:
                continue

            query = patch.reshape(1, -1).astype(np.float32)

            # Find best matching tile
            if best_k <= 1:
                idx, _ = find_nearest_tile(query, tile_matrix)
            else:
                idx = find_nearest_tile_topk(query, tile_matrix, best_k)

            # Place tile
            tile = tiles[idx]
            if tile.shape[0] != tile_h or tile.shape[1] != tile_w:
                tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            mosaic[x:x + tile_h, y:y + tile_w] = tile

            count += 1
            if count % 500 == 0 or count == total:
                print(f"  Placed {count}/{total} tiles...")

    # Trim to tiled area
    mosaic = mosaic[h_offset:h_offset + n_rows * tile_h,
                    w_offset:w_offset + n_cols * tile_w]

    # Opacity blend with original (trimmed to same region)
    if opacity > 0:
        target_trimmed = target_image[h_offset:h_offset + n_rows * tile_h,
                                       w_offset:w_offset + n_cols * tile_w]
        mosaic = cv2.addWeighted(target_trimmed, opacity, mosaic, 1 - opacity, 0)

    return mosaic


# ─── Target preparation ─────────────────────────────────────────────────────

def prepare_target(target_path, color_hex, black_hex):
    """Convert target image to monochrome color scheme, return as OpenCV BGR array."""
    with Image.open(target_path) as img:
        if img.mode in ('RGBA', 'LA', 'PA'):
            img = img.convert('RGB')
        gray = ImageOps.grayscale(img)
        mono = ImageOps.colorize(gray, black=black_hex, white=color_hex)
        mono = mono.convert('RGB')
        # PIL RGB -> OpenCV BGR
        return np.array(mono)[:, :, ::-1].copy()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Standalone Blue-In-Blue-Out mosaic generator (no faiss/dlib needed)"
    )

    parser.add_argument("--target", "-t", type=str, required=True,
                        help="Target image to recreate as a mosaic")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output path for the mosaic image (e.g. ./output/mosaic.jpg)")

    # Tile source options (pick one)
    tile_group = parser.add_mutually_exclusive_group()
    tile_group.add_argument("--codebook-dir", dest="codebook_dir", type=str,
                            help="Use existing tile images from this folder")
    tile_group.add_argument("--source-images", dest="source_images", type=str,
                            help="Convert real photos to monochrome tiles")
    tile_group.add_argument("--tile-mode", dest="tile_mode", type=str,
                            choices=["blocks", "halftone", "crosshatch", "noise"],
                            help="Generate synthetic tiles (default if nothing else specified)")

    # Generation settings
    parser.add_argument("--tile-count", dest="tile_count", type=int, default=50,
                        help="Number of synthetic tiles (default: 50)")
    parser.add_argument("--color", "-c", type=str, default="#0000FF",
                        help="Hex color for bright areas (default: #0000FF blue)")
    parser.add_argument("--black", "-b", dest="black_color", type=str, default="#000000",
                        help="Hex color for dark areas (default: #000000 black)")

    # Mosaic settings
    parser.add_argument("--scale", type=int, default=12,
                        help="Tile scale multiplier (default: 12)")
    parser.add_argument("--height-aspect", dest="height_aspect", type=float, default=4.0,
                        help="Height aspect (default: 4)")
    parser.add_argument("--width-aspect", dest="width_aspect", type=float, default=3.0,
                        help="Width aspect (default: 3)")
    parser.add_argument("--best-k", dest="best_k", type=int, default=1,
                        help="Pick from top K matches randomly (default: 1 = best match)")
    parser.add_argument("--opacity", type=float, default=0.0,
                        help="Blend original image on top (0.0 = pure mosaic, default: 0.0)")

    args = parser.parse_args()

    # Tile dimensions
    tile_h = int(args.height_aspect * args.scale)
    tile_w = int(args.width_aspect * args.scale)

    color_rgb = hex_to_rgb(args.color)
    black_rgb = hex_to_rgb(args.black_color)

    print("=" * 60)
    print("  BLUE MOSAIC - Standalone Generator")
    print("=" * 60)
    print(f"  Target:  {args.target}")
    print(f"  Output:  {args.output}")
    print(f"  Color:   {args.black_color} -> {args.color}")
    print(f"  Scale:   {args.scale} (tiles: {tile_w}x{tile_h} px)")
    print(f"  Aspect:  {args.height_aspect}:{args.width_aspect}")
    print()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # ─── Step 1: Get tiles ───────────────────────────────────────────────
    print("Step 1: Preparing tiles...")
    start = time.time()

    if args.codebook_dir:
        tiles = load_tiles_from_dir(args.codebook_dir, tile_h, tile_w)
    elif args.source_images:
        out_dir = os.path.join(os.path.dirname(args.output), "blue_tiles")
        tiles = convert_photos_to_blue(
            args.source_images, out_dir,
            args.color, args.black_color, tile_h, tile_w)
    else:
        # Default to synthetic blocks if nothing specified
        mode = args.tile_mode or "blocks"
        print(f"  Generating {args.tile_count} synthetic '{mode}' tiles...")
        gen_fn = TILE_GENERATORS[mode]
        tiles = gen_fn(color_rgb, black_rgb, args.tile_count, tile_w, tile_h)

    if not tiles:
        print("ERROR: No tiles available. Exiting.")
        sys.exit(1)

    print(f"  {len(tiles)} tiles ready ({time.time() - start:.1f}s)")

    # ─── Step 2: Prepare target ──────────────────────────────────────────
    print("\nStep 2: Preparing target image...")
    target_bgr = prepare_target(args.target, args.color, args.black_color)
    print(f"  Target size: {target_bgr.shape[1]}x{target_bgr.shape[0]} px")

    # Save the monochrome target for reference
    mono_path = os.path.splitext(args.output)[0] + '_target_mono.jpg'
    cv2.imwrite(mono_path, target_bgr)
    print(f"  Saved monochrome target: {mono_path}")

    # ─── Step 3: Build index and run mosaic ──────────────────────────────
    print("\nStep 3: Building mosaic...")
    start = time.time()

    tile_matrix = build_tile_index(tiles, tile_h, tile_w)
    print(f"  Tile index: {tile_matrix.shape[0]} tiles, {tile_matrix.shape[1]} dimensions")

    mosaic = mosaicify(
        target_bgr, tile_h, tile_w,
        tile_matrix, tiles,
        best_k=args.best_k,
        opacity=args.opacity,
    )

    elapsed = time.time() - start
    print(f"  Mosaic built in {elapsed:.1f}s")

    # ─── Step 4: Save ────────────────────────────────────────────────────
    cv2.imwrite(args.output, mosaic)
    print(f"\nDone! Mosaic saved to: {args.output}")
    print(f"  Size: {mosaic.shape[1]}x{mosaic.shape[0]} px")


if __name__ == '__main__':
    main()
