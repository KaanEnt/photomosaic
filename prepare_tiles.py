"""
prepare_tiles.py - "Blue-ifier" for photomosaic tiles

Converts a folder of source images into monochrome-tinted tiles.
Dark areas stay dark, bright areas become your chosen color.
Output is compatible with worldveil/photomosaic's mosaic.py (saves as .jpg).

Usage:
    conda activate Venv_001
    python prepare_tiles.py --input ./source_images --output ./blue_tiles
    python prepare_tiles.py --input ./source_images --output ./blue_tiles --color "#0044CC"
    python prepare_tiles.py --input ./source_images --output ./blue_tiles --color "#0044CC" --black "#000022"
"""

import os
import argparse
from PIL import Image, ImageOps


def hex_to_rgb(hex_color):
    """Convert a hex color string like '#0000FF' to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def make_monochrome(image_path, output_path, white_color, black_color):
    """
    Convert a single image to a monochrome-tinted version.
    
    1. Convert to grayscale (preserves luminosity/structure).
    2. Colorize: map black pixels -> black_color, white pixels -> white_color.
    3. Save as RGB JPEG for compatibility with photomosaic's *.jpg glob.
    """
    try:
        with Image.open(image_path) as img:
            # Handle images with alpha channel (PNG, WebP)
            if img.mode in ('RGBA', 'LA', 'PA'):
                img = img.convert('RGB')

            # Step 1: Grayscale (luminosity-based)
            gray = ImageOps.grayscale(img)

            # Step 2: Colorize — black areas become black_color, white areas become white_color
            mono = ImageOps.colorize(gray, black=black_color, white=white_color)

            # Step 3: Ensure RGB and save as JPEG
            mono = mono.convert('RGB')
            mono.save(output_path, 'JPEG', quality=95)
            print(f"  [OK] {os.path.basename(image_path)}")

    except Exception as e:
        print(f"  [SKIP] {os.path.basename(image_path)}: {e}")


def prepare_target(target_path, output_path, white_color, black_color):
    """
    Convert the target image to the same monochrome color scheme.
    This ensures the Faiss L2 matching works well — target and tiles
    share the same color space, so matching is purely structural.
    """
    print(f"\nPreparing target image: {target_path}")
    make_monochrome(target_path, output_path, white_color, black_color)
    print(f"Saved monochrome target to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert images to monochrome-tinted tiles for photomosaic"
    )
    parser.add_argument(
        "--input", "-i", dest="input_dir", type=str, required=True,
        help="Folder containing source images to convert"
    )
    parser.add_argument(
        "--output", "-o", dest="output_dir", type=str, required=True,
        help="Folder to save monochrome tiles into"
    )
    parser.add_argument(
        "--color", "-c", dest="color", type=str, default="#0000FF",
        help="Hex color for bright areas (default: #0000FF pure blue)"
    )
    parser.add_argument(
        "--black", "-b", dest="black_color", type=str, default="#000000",
        help="Hex color for dark areas (default: #000000 pure black)"
    )
    parser.add_argument(
        "--target", "-t", dest="target", type=str, default=None,
        help="Optional: also convert a target image to the same color scheme"
    )
    args = parser.parse_args()

    # Validate input
    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Input folder does not exist: {args.input_dir}")
        return

    # Create output folder
    os.makedirs(args.output_dir, exist_ok=True)

    white_color = args.color   # ImageOps.colorize accepts hex strings directly
    black_color = args.black_color

    print(f"=== Prepare Tiles ===")
    print(f"Input:  {os.path.abspath(args.input_dir)}")
    print(f"Output: {os.path.abspath(args.output_dir)}")
    print(f"Color:  dark={black_color} -> bright={white_color}")
    print()

    # Supported image extensions
    extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif')

    count = 0
    skipped = 0
    for filename in sorted(os.listdir(args.input_dir)):
        if filename.lower().endswith(extensions):
            input_path = os.path.join(args.input_dir, filename)

            # Always output as .jpg for photomosaic compatibility
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(args.output_dir, base_name + '.jpg')

            make_monochrome(input_path, output_path, white_color, black_color)
            count += 1
        else:
            skipped += 1

    print(f"\nConverted {count} images. Skipped {skipped} non-image files.")

    # Optionally convert the target image too
    if args.target:
        target_out = os.path.splitext(args.target)[0] + '_mono.jpg'
        prepare_target(args.target, target_out, white_color, black_color)

    print(f"\nDone. Point photomosaic's --codebook-dir to: {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    main()
