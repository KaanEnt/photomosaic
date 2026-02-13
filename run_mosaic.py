"""
run_mosaic.py - End-to-end "Blue In, Blue Out" mosaic pipeline

Orchestrates the full workflow:
  1. Prepare tiles (convert source images OR generate synthetic tiles)
  2. Prepare target image (convert to same monochrome color space)
  3. Run worldveil/photomosaic (mosaic.py)
  4. Apply final color tint to the finished mosaic (optional post-processing)

This script calls the other scripts and mosaic.py in sequence so you can
run the entire pipeline with a single command.

Usage:
    conda activate Venv_001

    # Using real photos as tiles (Option A: photo mosaic)
    python run_mosaic.py \
        --target tests/space.png \
        --source-images ./source_images \
        --color "#0000FF" \
        --scale 8

    # Using synthetic tiles (Option B: texture/block mosaic)
    python run_mosaic.py \
        --target tests/space.png \
        --synthetic blocks \
        --tile-count 30 \
        --color "#0000FF" \
        --scale 8

    # Skip mosaic step (just prepare tiles + target, run mosaic.py yourself)
    python run_mosaic.py \
        --target tests/space.png \
        --synthetic gradient \
        --color "#0044CC" \
        --prepare-only
"""

import os
import sys
import argparse
import subprocess
from PIL import Image, ImageOps


def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def apply_tint(image_path, output_path, color_hex):
    """
    Apply a color tint to a finished mosaic.
    Converts to grayscale then colorizes — useful if you ran the mosaic
    in grayscale and want to apply color as a final step.
    """
    print(f"\n=== Applying color tint ===")
    print(f"Input:  {image_path}")
    print(f"Color:  {color_hex}")

    with Image.open(image_path) as img:
        gray = ImageOps.grayscale(img.convert('RGB'))
        tinted = ImageOps.colorize(gray, black="black", white=color_hex)
        tinted = tinted.convert('RGB')
        tinted.save(output_path, quality=95)
        print(f"Saved:  {output_path}")


def run_command(cmd, description):
    """Run a subprocess command with error handling."""
    print(f"\n{'='*50}")
    print(f"  {description}")
    print(f"{'='*50}")
    print(f"  $ {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with exit code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end Blue-In-Blue-Out mosaic pipeline"
    )

    # Target image
    parser.add_argument(
        "--target", "-t", type=str, required=True,
        help="Target image to recreate as a mosaic"
    )

    # Tile source: real photos OR synthetic
    tile_group = parser.add_mutually_exclusive_group(required=True)
    tile_group.add_argument(
        "--source-images", dest="source_images", type=str,
        help="Folder of real photos to use as tiles (Option A)"
    )
    tile_group.add_argument(
        "--synthetic", dest="synthetic", type=str,
        choices=["gradient", "blocks", "halftone", "all"],
        help="Generate synthetic tiles instead (Option B)"
    )

    # Color settings
    parser.add_argument(
        "--color", "-c", type=str, default="#0000FF",
        help="Hex color for bright areas (default: #0000FF blue)"
    )
    parser.add_argument(
        "--black", "-b", dest="black_color", type=str, default="#000000",
        help="Hex color for dark areas (default: #000000 black)"
    )

    # Mosaic settings
    parser.add_argument("--scale", type=int, default=8, help="Tile scale (default: 8)")
    parser.add_argument("--height-aspect", dest="height_aspect", type=float, default=4.0)
    parser.add_argument("--width-aspect", dest="width_aspect", type=float, default=3.0)
    parser.add_argument("--opacity", type=float, default=0.0, help="Opacity overlay (default: 0.0)")
    parser.add_argument("--best-k", dest="best_k", type=int, default=1, help="Top-K tile selection")
    parser.add_argument("--tile-count", dest="tile_count", type=int, default=20,
                        help="Number of synthetic tiles to generate (default: 20)")

    # Output settings
    parser.add_argument(
        "--output-dir", dest="output_dir", type=str, default="./output",
        help="Directory for all output files (default: ./output)"
    )
    parser.add_argument(
        "--prepare-only", dest="prepare_only", action="store_true", default=False,
        help="Only prepare tiles and target; don't run mosaic.py"
    )
    parser.add_argument(
        "--tint-after", dest="tint_after", action="store_true", default=False,
        help="Apply color tint as a post-processing step on the finished mosaic"
    )

    args = parser.parse_args()

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    tiles_dir = os.path.join(args.output_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    target_basename = os.path.splitext(os.path.basename(args.target))[0]
    mono_target = os.path.join(args.output_dir, f"{target_basename}_mono.jpg")

    print("=" * 60)
    print("  BLUE-IN-BLUE-OUT MOSAIC PIPELINE")
    print("=" * 60)
    print(f"  Target:     {args.target}")
    print(f"  Color:      {args.black_color} -> {args.color}")
    print(f"  Scale:      {args.scale}")
    print(f"  Aspect:     {args.height_aspect}:{args.width_aspect}")
    print(f"  Output:     {os.path.abspath(args.output_dir)}")
    if args.source_images:
        print(f"  Tile mode:  Real photos from {args.source_images}")
    else:
        print(f"  Tile mode:  Synthetic ({args.synthetic}), {args.tile_count} tiles")
    print()

    # ─── Step 1: Prepare tiles ───────────────────────────────────────────
    if args.source_images:
        run_command([
            sys.executable, "prepare_tiles.py",
            "--input", args.source_images,
            "--output", tiles_dir,
            "--color", args.color,
            "--black", args.black_color,
        ], "Step 1: Converting source images to monochrome tiles")
    else:
        run_command([
            sys.executable, "generate_synthetic_tiles.py",
            "--output", tiles_dir,
            "--mode", args.synthetic,
            "--count", str(args.tile_count),
            "--color", args.color,
            "--black", args.black_color,
            "--height-aspect", str(args.height_aspect),
            "--width-aspect", str(args.width_aspect),
            "--scale", str(args.scale),
        ], "Step 1: Generating synthetic tiles")

    # ─── Step 2: Prepare target image ────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Step 2: Converting target to monochrome")
    print(f"{'='*50}")

    try:
        with Image.open(args.target) as img:
            if img.mode in ('RGBA', 'LA', 'PA'):
                img = img.convert('RGB')
            gray = ImageOps.grayscale(img)
            mono = ImageOps.colorize(gray, black=args.black_color, white=args.color)
            mono = mono.convert('RGB')
            mono.save(mono_target, 'JPEG', quality=95)
            print(f"  Saved monochrome target: {mono_target}")
    except Exception as e:
        print(f"  ERROR converting target: {e}")
        sys.exit(1)

    if args.prepare_only:
        print(f"\n{'='*50}")
        print(f"  --prepare-only: Stopping here.")
        print(f"{'='*50}")
        print(f"\n  Tiles ready at:  {os.path.abspath(tiles_dir)}")
        print(f"  Target ready at: {os.path.abspath(mono_target)}")
        print(f"\n  Run mosaic.py manually:")
        print(f"    python mosaic.py \\")
        print(f"      --target \"{mono_target}\" \\")
        print(f"      --codebook-dir \"{tiles_dir}\" \\")
        print(f"      --savepath \"{args.output_dir}/%s-mosaic-scale-%d.jpg\" \\")
        print(f"      --scale {args.scale} \\")
        print(f"      --height-aspect {args.height_aspect} \\")
        print(f"      --width-aspect {args.width_aspect}")
        return

    # ─── Step 3: Run photomosaic ─────────────────────────────────────────
    mosaic_savepath = os.path.join(args.output_dir, "%s-mosaic-scale-%d.jpg")
    run_command([
        sys.executable, "mosaic.py",
        "--target", mono_target,
        "--codebook-dir", tiles_dir,
        "--savepath", mosaic_savepath,
        "--scale", str(args.scale),
        "--height-aspect", str(args.height_aspect),
        "--width-aspect", str(args.width_aspect),
        "--opacity", str(args.opacity),
        "--best-k", str(args.best_k),
    ], "Step 3: Running photomosaic")

    # ─── Step 4 (optional): Apply tint to finished mosaic ────────────────
    if args.tint_after:
        mosaic_output = mosaic_savepath % (target_basename + "_mono", args.scale)
        tinted_output = os.path.join(
            args.output_dir,
            f"{target_basename}-tinted-scale-{args.scale}.jpg"
        )
        if os.path.exists(mosaic_output):
            apply_tint(mosaic_output, tinted_output, args.color)
        else:
            print(f"\nWARNING: Expected mosaic output not found at {mosaic_output}")
            print("  Skipping tint step. You can apply it manually with:")
            print(f"    python -c \"from run_mosaic import apply_tint; apply_tint('{mosaic_output}', '{tinted_output}', '{args.color}')\"")

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Output directory: {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    main()
