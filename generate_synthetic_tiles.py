"""
generate_synthetic_tiles.py - Create synthetic tile datasets for photomosaic

Generates tiles with VISIBLE internal texture/patterns at varying brightness
levels. The Faiss engine matches tiles by L2 pixel distance, so each tile
needs distinct internal structure — otherwise the mosaic looks like a smooth
tinted photo with no visible tile grid.

Modes:
  - "blocks":     Pixel-block grid patterns with varying coverage (digital/retro look)
  - "halftone":   Centered dots of varying radius (classic print halftone)
  - "crosshatch": Line-based hatching patterns (engraving/sketch look)
  - "noise":      Dithered noise patterns at varying densities (stipple look)
  - "all":        Generate all modes into the same folder

Each mode produces tiles from near-black to near-full-color with visible
internal pattern. Tiles are saved as .jpg for photomosaic compatibility.

Usage:
    conda activate Venv_001

    python generate_synthetic_tiles.py --output ./blue_tiles --mode blocks --count 50
    python generate_synthetic_tiles.py --output ./blue_tiles --mode all --count 40
    python generate_synthetic_tiles.py --output ./tiles --mode blocks --color "#0044CC" --height-aspect 4 --width-aspect 3 --scale 16
"""

import os
import argparse
import random
import math
import numpy as np
from PIL import Image, ImageDraw


def hex_to_rgb(hex_color):
    """Convert '#RRGGBB' to (R, G, B) tuple."""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def lerp_color(c1, c2, t):
    """Linearly interpolate between two RGB tuples by factor t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


# ─── Block patterns ─────────────────────────────────────────────────────────

def generate_block_tiles(output_dir, color_rgb, black_rgb, count, tile_w, tile_h):
    """
    Pixel-block grid: a fixed grid of cells, each cell either filled or empty.
    Brightness is controlled by how many cells are filled.
    At low brightness only a few scattered cells are on; at high brightness
    nearly all cells are on. This gives a visible chunky pixel-grid texture.
    """
    print(f"Generating {count} block tiles ({tile_w}x{tile_h})...")

    # Use a grid that's visible: aim for cells ~4-8px, at least 3x3 grid
    cells_x = max(3, tile_w // 6)
    cells_y = max(3, tile_h // 6)
    cell_w = tile_w / cells_x
    cell_h = tile_h / cells_y
    total_cells = cells_x * cells_y

    for i in range(count):
        t = i / max(count - 1, 1)  # 0.0 to 1.0

        img = Image.new('RGB', (tile_w, tile_h), black_rgb)
        draw = ImageDraw.Draw(img)

        # Number of cells to fill — proportional to brightness
        n_filled = int(total_cells * t)

        # Deterministic seed per tile so results are reproducible
        rng = random.Random(42 + i)
        all_cells = [(cx, cy) for cy in range(cells_y) for cx in range(cells_x)]
        rng.shuffle(all_cells)
        filled_cells = set(tuple(c) for c in all_cells[:n_filled])

        # Color intensity also scales with brightness
        fill = lerp_color(black_rgb, color_rgb, 0.4 + 0.6 * t)

        for cy in range(cells_y):
            for cx in range(cells_x):
                if (cx, cy) in filled_cells:
                    x0 = int(cx * cell_w)
                    y0 = int(cy * cell_h)
                    x1 = int((cx + 1) * cell_w) - 1  # -1 for gap between cells
                    y1 = int((cy + 1) * cell_h) - 1
                    draw.rectangle([x0, y0, x1, y1], fill=fill)

        path = os.path.join(output_dir, f"blocks_{i+1:03d}.jpg")
        img.save(path, 'JPEG', quality=95)
        print(f"  [OK] blocks_{i+1:03d}.jpg  fill={n_filled}/{total_cells} ({t:.0%})")


# ─── Halftone dot patterns ──────────────────────────────────────────────────

def generate_halftone_tiles(output_dir, color_rgb, black_rgb, count, tile_w, tile_h):
    """
    Classic halftone: a grid of dots where dot radius controls brightness.
    Small dots = dark, large dots = bright. The dot grid is always visible.
    """
    print(f"Generating {count} halftone tiles ({tile_w}x{tile_h})...")

    # Grid of dots — aim for 3x4 or 4x5 dots per tile so they're visible
    dots_x = max(2, tile_w // 12)
    dots_y = max(2, tile_h // 12)
    spacing_x = tile_w / dots_x
    spacing_y = tile_h / dots_y
    max_radius = min(spacing_x, spacing_y) * 0.48  # max radius before dots touch

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
                    draw.ellipse(
                        [cx - radius, cy - radius, cx + radius, cy + radius],
                        fill=fill
                    )

        path = os.path.join(output_dir, f"halftone_{i+1:03d}.jpg")
        img.save(path, 'JPEG', quality=95)
        print(f"  [OK] halftone_{i+1:03d}.jpg  dot_radius={t:.2f}")


# ─── Crosshatch / line patterns ─────────────────────────────────────────────

def generate_crosshatch_tiles(output_dir, color_rgb, black_rgb, count, tile_w, tile_h):
    """
    Line-based hatching: at low brightness, sparse diagonal lines.
    As brightness increases, add more line directions and density,
    building up to a dense crosshatch that approaches solid color.
    """
    print(f"Generating {count} crosshatch tiles ({tile_w}x{tile_h})...")

    for i in range(count):
        t = i / max(count - 1, 1)

        img = Image.new('RGB', (tile_w, tile_h), black_rgb)
        draw = ImageDraw.Draw(img)

        if t < 0.02:
            pass  # pure black
        elif t > 0.95:
            # Near-solid: fill with color
            img = Image.new('RGB', (tile_w, tile_h), color_rgb)
        else:
            fill = lerp_color(black_rgb, color_rgb, 0.5 + 0.5 * t)
            line_width = max(1, int(1 + t * 2))

            # Line spacing decreases (denser) as brightness increases
            spacing = max(3, int(20 * (1 - t) + 3))
            diag = int(math.sqrt(tile_w**2 + tile_h**2))

            # Layer 1: diagonal lines (always present for t > 0)
            for offset in range(-diag, diag, spacing):
                draw.line([(offset, 0), (offset + tile_h, tile_h)],
                          fill=fill, width=line_width)

            # Layer 2: opposite diagonal (adds at ~30% brightness)
            if t > 0.3:
                for offset in range(-diag, diag, spacing):
                    draw.line([(tile_w - offset, 0), (tile_w - offset - tile_h, tile_h)],
                              fill=fill, width=line_width)

            # Layer 3: horizontal lines (adds at ~60% brightness)
            if t > 0.6:
                for y in range(0, tile_h, spacing):
                    draw.line([(0, y), (tile_w, y)], fill=fill, width=line_width)

            # Layer 4: vertical lines (adds at ~80% brightness)
            if t > 0.8:
                for x in range(0, tile_w, spacing):
                    draw.line([(x, 0), (x, tile_h)], fill=fill, width=line_width)

        path = os.path.join(output_dir, f"crosshatch_{i+1:03d}.jpg")
        img.save(path, 'JPEG', quality=95)
        print(f"  [OK] crosshatch_{i+1:03d}.jpg  density={t:.2f}")


# ─── Noise / dither patterns ────────────────────────────────────────────────

def generate_noise_tiles(output_dir, color_rgb, black_rgb, count, tile_w, tile_h):
    """
    Dithered noise: random pixels are either on (color) or off (black).
    The fraction of 'on' pixels controls brightness. Gives a stipple/grain look.
    """
    print(f"Generating {count} noise tiles ({tile_w}x{tile_h})...")

    for i in range(count):
        t = i / max(count - 1, 1)

        # Create a random mask — fraction t of pixels are "on"
        rng = np.random.RandomState(seed=42 + i)
        mask = rng.random((tile_h, tile_w)) < t

        # Build RGB image: on-pixels get color, off-pixels get black
        arr = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        fill = lerp_color(black_rgb, color_rgb, 0.5 + 0.5 * t)
        arr[mask] = fill
        arr[~mask] = black_rgb

        img = Image.fromarray(arr, 'RGB')
        path = os.path.join(output_dir, f"noise_{i+1:03d}.jpg")
        img.save(path, 'JPEG', quality=95)
        print(f"  [OK] noise_{i+1:03d}.jpg  density={t:.2f}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic tile datasets with visible texture for photomosaic"
    )
    parser.add_argument(
        "--output", "-o", dest="output_dir", type=str, required=True,
        help="Folder to save generated tiles"
    )
    parser.add_argument(
        "--mode", "-m", dest="mode", type=str, default="blocks",
        choices=["blocks", "halftone", "crosshatch", "noise", "all"],
        help="Tile pattern mode (default: blocks)"
    )
    parser.add_argument(
        "--count", "-n", dest="count", type=int, default=50,
        help="Number of tiles to generate per mode (default: 50)"
    )
    parser.add_argument(
        "--color", "-c", dest="color", type=str, default="#0000FF",
        help="Hex color for bright areas (default: #0000FF blue)"
    )
    parser.add_argument(
        "--black", "-b", dest="black_color", type=str, default="#000000",
        help="Hex color for dark areas (default: #000000 black)"
    )
    parser.add_argument(
        "--height-aspect", dest="height_aspect", type=float, default=4.0,
        help="Height aspect ratio (default: 4)"
    )
    parser.add_argument(
        "--width-aspect", dest="width_aspect", type=float, default=3.0,
        help="Width aspect ratio (default: 3)"
    )
    parser.add_argument(
        "--scale", dest="scale", type=int, default=16,
        help="Scale multiplier for tile dimensions (default: 16)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    color_rgb = hex_to_rgb(args.color)
    black_rgb = hex_to_rgb(args.black_color)
    tile_h = int(args.height_aspect * args.scale)
    tile_w = int(args.width_aspect * args.scale)

    print(f"=== Generate Synthetic Tiles ===")
    print(f"Output: {os.path.abspath(args.output_dir)}")
    print(f"Mode:   {args.mode}")
    print(f"Count:  {args.count} tiles per mode")
    print(f"Color:  dark={args.black_color} -> bright={args.color}")
    print(f"Tile:   {tile_w}x{tile_h} px (aspect {args.height_aspect}:{args.width_aspect}, scale {args.scale})")
    print()

    generators = {
        "blocks": generate_block_tiles,
        "halftone": generate_halftone_tiles,
        "crosshatch": generate_crosshatch_tiles,
        "noise": generate_noise_tiles,
    }

    modes = list(generators.keys()) if args.mode == "all" else [args.mode]
    for mode in modes:
        generators[mode](args.output_dir, color_rgb, black_rgb, args.count, tile_w, tile_h)
        print()

    total = len([f for f in os.listdir(args.output_dir) if f.endswith('.jpg')])
    print(f"Done. {total} tiles in {os.path.abspath(args.output_dir)}")
    print(f"Point photomosaic's --codebook-dir to this folder.")


if __name__ == '__main__':
    main()
