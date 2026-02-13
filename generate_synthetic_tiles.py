"""
generate_synthetic_tiles.py - Create synthetic tile datasets for photomosaic

Generates tiles with varying brightness/density patterns so the Faiss engine
matches structure (luminosity) rather than photo content. This produces a
"halftone" or "pixel block" effect in the final mosaic.

Three modes:
  - "gradient":  Solid color tiles from black to full color (N brightness steps)
  - "blocks":    Grid-of-blocks patterns with varying density/size
  - "halftone":  Dot patterns with varying radius (classic halftone look)

All tiles are saved as .jpg at the specified aspect ratio for photomosaic compatibility.

Usage:
    conda activate Venv_001

    # Simple gradient tiles (10 shades of blue)
    python generate_synthetic_tiles.py --output ./blue_tiles --mode gradient --count 20

    # Block-density tiles (pixel block look)
    python generate_synthetic_tiles.py --output ./blue_tiles --mode blocks --count 20

    # Halftone dot tiles
    python generate_synthetic_tiles.py --output ./blue_tiles --mode halftone --count 20

    # Custom color and aspect ratio
    python generate_synthetic_tiles.py --output ./tiles --mode blocks --color "#0044CC" --height-aspect 4 --width-aspect 3 --scale 16
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw


def hex_to_rgb(hex_color):
    """Convert '#RRGGBB' to (R, G, B) tuple."""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def lerp_color(black_rgb, color_rgb, t):
    """Linearly interpolate between black_rgb and color_rgb by factor t in [0, 1]."""
    return tuple(int(b + (c - b) * t) for b, c in zip(black_rgb, color_rgb))


def generate_gradient_tiles(output_dir, color_rgb, black_rgb, count, tile_w, tile_h):
    """
    Generate solid-color tiles at evenly spaced brightness levels.
    tile_01 = pure black, tile_N = full color.
    """
    print(f"Generating {count} gradient tiles...")
    for i in range(count):
        t = i / max(count - 1, 1)  # 0.0 to 1.0
        fill = lerp_color(black_rgb, color_rgb, t)
        img = Image.new('RGB', (tile_w, tile_h), fill)
        path = os.path.join(output_dir, f"gradient_{i+1:03d}.jpg")
        img.save(path, 'JPEG', quality=95)
        print(f"  [OK] gradient_{i+1:03d}.jpg  brightness={t:.2f}  color={fill}")


def generate_block_tiles(output_dir, color_rgb, black_rgb, count, tile_w, tile_h):
    """
    Generate tiles with grid-of-blocks patterns at varying densities.
    Low index = sparse small blocks (dark), high index = dense large blocks (bright).
    """
    print(f"Generating {count} block-density tiles...")
    for i in range(count):
        t = i / max(count - 1, 1)

        # Background is black, blocks are colored
        img = Image.new('RGB', (tile_w, tile_h), black_rgb)
        draw = ImageDraw.Draw(img)

        if t < 0.02:
            # Nearly black — leave as-is
            pass
        elif t > 0.98:
            # Nearly full color — solid fill
            img = Image.new('RGB', (tile_w, tile_h), color_rgb)
        else:
            # Determine grid size: more blocks as brightness increases
            # Grid cells range from 2x2 to 8x8
            grid_n = max(2, int(2 + 6 * t))
            cell_w = tile_w / grid_n
            cell_h = tile_h / grid_n

            # Block size within each cell scales with brightness
            block_fraction = 0.3 + 0.65 * t  # 30% to 95% of cell

            fill = lerp_color(black_rgb, color_rgb, 0.5 + 0.5 * t)

            for row in range(grid_n):
                for col in range(grid_n):
                    cx = col * cell_w + cell_w / 2
                    cy = row * cell_h + cell_h / 2
                    bw = cell_w * block_fraction / 2
                    bh = cell_h * block_fraction / 2
                    draw.rectangle(
                        [cx - bw, cy - bh, cx + bw, cy + bh],
                        fill=fill
                    )

        path = os.path.join(output_dir, f"blocks_{i+1:03d}.jpg")
        img.save(path, 'JPEG', quality=95)
        print(f"  [OK] blocks_{i+1:03d}.jpg  density={t:.2f}")


def generate_halftone_tiles(output_dir, color_rgb, black_rgb, count, tile_w, tile_h):
    """
    Generate tiles with centered dot patterns of varying radius (halftone effect).
    Small dot = dark tile, large dot = bright tile.
    """
    print(f"Generating {count} halftone tiles...")
    for i in range(count):
        t = i / max(count - 1, 1)

        img = Image.new('RGB', (tile_w, tile_h), black_rgb)
        draw = ImageDraw.Draw(img)

        if t < 0.02:
            pass  # pure black
        elif t > 0.98:
            img = Image.new('RGB', (tile_w, tile_h), color_rgb)
        else:
            # Dot radius scales with brightness
            max_radius = min(tile_w, tile_h) / 2
            radius = max_radius * t
            cx, cy = tile_w / 2, tile_h / 2

            fill = lerp_color(black_rgb, color_rgb, 0.6 + 0.4 * t)
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=fill
            )

        path = os.path.join(output_dir, f"halftone_{i+1:03d}.jpg")
        img.save(path, 'JPEG', quality=95)
        print(f"  [OK] halftone_{i+1:03d}.jpg  radius_fraction={t:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic tile datasets for photomosaic"
    )
    parser.add_argument(
        "--output", "-o", dest="output_dir", type=str, required=True,
        help="Folder to save generated tiles"
    )
    parser.add_argument(
        "--mode", "-m", dest="mode", type=str, default="gradient",
        choices=["gradient", "blocks", "halftone", "all"],
        help="Tile generation mode (default: gradient)"
    )
    parser.add_argument(
        "--count", "-n", dest="count", type=int, default=20,
        help="Number of tiles to generate per mode (default: 20)"
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
        "gradient": generate_gradient_tiles,
        "blocks": generate_block_tiles,
        "halftone": generate_halftone_tiles,
    }

    modes = list(generators.keys()) if args.mode == "all" else [args.mode]
    for mode in modes:
        generators[mode](args.output_dir, color_rgb, black_rgb, args.count, tile_w, tile_h)
        print()

    total = len(os.listdir(args.output_dir))
    print(f"Done. {total} tiles in {os.path.abspath(args.output_dir)}")
    print(f"Point photomosaic's --codebook-dir to this folder.")


if __name__ == '__main__':
    main()
