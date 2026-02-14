"""
blue_mosaic.py - Standalone "Blue In, Blue Out" mosaic generator

Creates photomosaics from images, GIFs, and videos using monochrome-tinted
tiles. Self-contained — only needs Pillow, OpenCV, and NumPy (no faiss,
dlib, or emosaic).

How it works:
  1. Generate (or load) a set of tiles at varying brightness levels, each
     with a visible internal texture pattern.
  2. Convert the target to grayscale.
  3. Divide into a grid of cells.
  4. For each cell, compute average brightness and pick the closest tile.
  5. Place tiles into the output mosaic.

For GIFs/videos, this is done per-frame and reassembled.

Usage:
    conda activate Venv_001

    # ── Still images ──
    python blue_mosaic.py --target tests/space.png --output ./output/space_mosaic.jpg --tile-mode blocks --scale 6

    # ── GIF ──
    python blue_mosaic.py --target tests/moon.gif --output ./output/moon_mosaic.gif --tile-mode blocks --scale 6 --color "#0072CE"

    # ── Video (mp4) ──
    python blue_mosaic.py --target myvideo.mp4 --output ./output/mosaic_video.mp4 --tile-mode halftone --scale 8

    # ── Other options ──
    python blue_mosaic.py --target tests/space.png --output ./output/space_fine.jpg --tile-mode blocks --levels 128 --scale 4 --opacity 0.2 --best-k 3
    python blue_mosaic.py --target tests/space.png --output ./output/space_photos.jpg --source-images ./family_photos --scale 8
    python blue_mosaic.py --target tests/space.png --output ./output/space_custom.jpg --codebook-dir ./my_tiles --scale 8
"""

import os
import sys
import glob
import time
import argparse
import random
import math

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


def add_alpha_from_brightness(bgr_tile, black_rgb):
    """
    Convert a 3-channel BGR tile to 4-channel BGRA where pixels matching
    the black/background color become fully transparent (alpha=0) and all
    other pixels (the colored pattern) stay fully opaque (alpha=255).
    A small tolerance is used so near-black pixels from anti-aliasing
    also become transparent.
    """
    # black_rgb is in RGB order; convert to BGR for comparison
    black_bgr = np.array([black_rgb[2], black_rgb[1], black_rgb[0]], dtype=np.float32)
    tile_f = bgr_tile.astype(np.float32)
    # Per-pixel L2 distance from the black color
    dist = np.sqrt(np.sum((tile_f - black_bgr) ** 2, axis=2))
    # Pixels within a small tolerance of the background color -> transparent
    # Everything else -> fully opaque
    tolerance = 10.0
    alpha = np.where(dist <= tolerance, 0, 255).astype(np.uint8)
    # Build BGRA
    bgra = np.dstack([bgr_tile, alpha])
    return bgra


# ═══════════════════════════════════════════════════════════════════════════════
#  TILE GENERATORS
#
#  Each generator produces `count` tiles at evenly-spaced brightness levels.
#  Tile at index 0 is darkest (near black), tile at index count-1 is brightest.
#  Each tile has visible internal structure so the mosaic grid is apparent.
# ═══════════════════════════════════════════════════════════════════════════════

def generate_tiles_blocks(color_rgb, black_rgb, count, tile_w, tile_h, transparent=False):
    """
    Ordered pixel-block grid. Cells fill from top-left in a consistent
    scan order (not random), so brightness increases smoothly and the
    pattern is structured, not noisy.
    """
    tiles = []
    # Grid: ~4-6px per cell so blocks are visible
    cells_x = max(3, tile_w // 5)
    cells_y = max(3, tile_h // 5)
    cell_w = tile_w / cells_x
    cell_h = tile_h / cells_y
    total_cells = cells_x * cells_y

    # Build a fixed fill order: diagonal sweep for visual interest
    cell_order = []
    for diag in range(cells_x + cells_y - 1):
        for cy in range(cells_y):
            cx = diag - cy
            if 0 <= cx < cells_x:
                cell_order.append((cx, cy))

    for i in range(count):
        t = i / max(count - 1, 1)
        img = Image.new('RGB', (tile_w, tile_h), black_rgb)
        draw = ImageDraw.Draw(img)

        n_filled = int(total_cells * t)
        fill = lerp_color(black_rgb, color_rgb, max(0.3, t))

        for idx in range(n_filled):
            if idx >= len(cell_order):
                break
            cx, cy = cell_order[idx]
            x0 = int(cx * cell_w)
            y0 = int(cy * cell_h)
            x1 = int((cx + 1) * cell_w) - 1
            y1 = int((cy + 1) * cell_h) - 1
            draw.rectangle([x0, y0, x1, y1], fill=fill)

        arr = np.array(img)[:, :, ::-1]
        if transparent:
            arr = add_alpha_from_brightness(arr, black_rgb)
        tiles.append(arr)
    return tiles


def generate_tiles_halftone(color_rgb, black_rgb, count, tile_w, tile_h, transparent=False):
    """
    Single centered dot per tile. Dot radius scales with brightness.
    Clean, classic halftone look.
    """
    tiles = []
    max_radius = min(tile_w, tile_h) / 2.0

    for i in range(count):
        t = i / max(count - 1, 1)
        img = Image.new('RGB', (tile_w, tile_h), black_rgb)
        draw = ImageDraw.Draw(img)

        if t > 0.005:
            radius = max(1, max_radius * math.sqrt(t))  # sqrt for perceptual scaling
            cx, cy = tile_w / 2.0, tile_h / 2.0
            fill = lerp_color(black_rgb, color_rgb, max(0.4, t))
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=fill
            )

        arr = np.array(img)[:, :, ::-1]
        if transparent:
            arr = add_alpha_from_brightness(arr, black_rgb)
        tiles.append(arr)
    return tiles


def generate_tiles_crosshatch(color_rgb, black_rgb, count, tile_w, tile_h, transparent=False):
    """
    Line hatching with increasing density. Layers build up:
    sparse diagonals -> cross-diagonals -> horizontal -> vertical.
    """
    tiles = []
    for i in range(count):
        t = i / max(count - 1, 1)
        img = Image.new('RGB', (tile_w, tile_h), black_rgb)

        if t > 0.98:
            img = Image.new('RGB', (tile_w, tile_h), color_rgb)
        elif t > 0.01:
            draw = ImageDraw.Draw(img)
            fill = lerp_color(black_rgb, color_rgb, max(0.4, t))
            lw = max(1, int(1 + t * 2))
            spacing = max(3, int(18 * (1 - t) + 3))
            diag = int(math.sqrt(tile_w**2 + tile_h**2))

            for off in range(-diag, diag, spacing):
                draw.line([(off, 0), (off + tile_h, tile_h)], fill=fill, width=lw)
            if t > 0.25:
                for off in range(-diag, diag, spacing):
                    draw.line([(tile_w - off, 0), (tile_w - off - tile_h, tile_h)], fill=fill, width=lw)
            if t > 0.55:
                for y in range(0, tile_h, spacing):
                    draw.line([(0, y), (tile_w, y)], fill=fill, width=lw)
            if t > 0.80:
                for x in range(0, tile_w, spacing):
                    draw.line([(x, 0), (x, tile_h)], fill=fill, width=lw)

        arr = np.array(img)[:, :, ::-1]
        if transparent:
            arr = add_alpha_from_brightness(arr, black_rgb)
        tiles.append(arr)
    return tiles


def generate_tiles_noise(color_rgb, black_rgb, count, tile_w, tile_h, transparent=False):
    """
    Ordered dithering (Bayer matrix style) — not random noise.
    Produces a structured, repeatable dither pattern at each brightness level.
    """
    tiles = []

    # Build a Bayer-like threshold matrix tiled to cover the tile
    bayer_2x2 = np.array([[0, 2], [3, 1]], dtype=np.float32) / 4.0
    # Tile it to cover the full tile dimensions
    reps_y = math.ceil(tile_h / 2)
    reps_x = math.ceil(tile_w / 2)
    threshold = np.tile(bayer_2x2, (reps_y, reps_x))[:tile_h, :tile_w]

    for i in range(count):
        t = i / max(count - 1, 1)
        mask = threshold < t

        arr = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        fill = lerp_color(black_rgb, color_rgb, max(0.4, t))
        arr[mask] = fill
        arr[~mask] = black_rgb
        bgr = arr[:, :, ::-1].copy()  # RGB -> BGR
        if transparent:
            bgr = add_alpha_from_brightness(bgr, black_rgb)
        tiles.append(bgr)
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
    """Convert real photos to monochrome, resize to tile dims."""
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

                out_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.jpg')
                mono.save(out_path, 'JPEG', quality=95)

                arr = np.array(mono)[:, :, ::-1]
                resized = cv2.resize(arr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                tiles.append(resized)
        except Exception as e:
            print(f"  [SKIP] {filename}: {e}")

    print(f"  Converted {len(tiles)} photos to monochrome tiles")
    return tiles


# ═══════════════════════════════════════════════════════════════════════════════
#  MOSAIC ENGINE
#
#  Brightness-based matching: compute average brightness per tile and per
#  target cell, then pick the tile whose brightness is closest. This is
#  O(1) per cell (binary search into sorted brightness list) instead of
#  O(N*D) brute-force L2.
# ═══════════════════════════════════════════════════════════════════════════════

def compute_tile_brightness(tiles):
    """Compute average brightness (0-255) for each tile."""
    brightnesses = []
    for tile in tiles:
        # Handle both BGR (3-ch) and BGRA (4-ch) tiles
        bgr = tile[:, :, :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        brightnesses.append(float(np.mean(gray)))
    return np.array(brightnesses)


def build_brightness_index(brightnesses):
    """Sort tiles by brightness for fast lookup."""
    order = np.argsort(brightnesses)
    sorted_brightness = brightnesses[order]
    return order, sorted_brightness


def find_best_tile(target_brightness, sorted_brightness, sort_order, best_k=1):
    """Find the tile(s) whose brightness best matches the target."""
    # Binary search for closest brightness
    idx = np.searchsorted(sorted_brightness, target_brightness)
    idx = min(idx, len(sorted_brightness) - 1)

    if best_k <= 1:
        # Check idx and idx-1, return the closer one
        if idx > 0:
            d_left = abs(sorted_brightness[idx - 1] - target_brightness)
            d_right = abs(sorted_brightness[idx] - target_brightness)
            if d_left < d_right:
                idx = idx - 1
        return sort_order[idx]
    else:
        # Return random pick from top-k nearest
        candidates = []
        lo = max(0, idx - best_k)
        hi = min(len(sorted_brightness), idx + best_k + 1)
        for j in range(lo, hi):
            candidates.append((abs(sorted_brightness[j] - target_brightness), sort_order[j]))
        candidates.sort(key=lambda x: x[0])
        top = candidates[:best_k]
        return random.choice(top)[1]


def mosaicify(target_gray, tile_h, tile_w, tiles, sorted_brightness, sort_order,
              best_k=1, opacity=0.0, target_color=None, transparent=False, verbose=True):
    """
    Build the mosaic by matching each grid cell's average brightness
    to the best tile.
    """
    img_h, img_w = target_gray.shape

    n_rows = img_h // tile_h
    n_cols = img_w // tile_w
    h_offset = (img_h % tile_h) // 2
    w_offset = (img_w % tile_w) // 2

    # Output mosaic: BGRA (4 channels) when transparent, BGR (3 channels) otherwise
    out_h = n_rows * tile_h
    out_w = n_cols * tile_w
    n_channels = 4 if transparent else 3
    mosaic = np.zeros((out_h, out_w, n_channels), dtype=np.uint8)

    total = n_rows * n_cols

    if verbose:
        print(f"  Grid: {n_cols} x {n_rows} = {total} tiles")
        print(f"  Output: {out_w} x {out_h} px")

    for row in range(n_rows):
        for col in range(n_cols):
            src_x = row * tile_h + h_offset
            src_y = col * tile_w + w_offset

            patch = target_gray[src_x:src_x + tile_h, src_y:src_y + tile_w]
            if patch.shape[0] != tile_h or patch.shape[1] != tile_w:
                continue
            avg_brightness = float(np.mean(patch))

            tile_idx = find_best_tile(avg_brightness, sorted_brightness, sort_order, best_k)
            tile = tiles[tile_idx]

            out_x = row * tile_h
            out_y = col * tile_w
            mosaic[out_x:out_x + tile_h, out_y:out_y + tile_w] = tile

    # Opacity blend with the monochrome target (skip when transparent)
    if opacity > 0 and target_color is not None and not transparent:
        target_crop = target_color[h_offset:h_offset + out_h, w_offset:w_offset + out_w]
        mosaic = cv2.addWeighted(target_crop, opacity, mosaic, 1 - opacity, 0)

    return mosaic


# ─── Frame conversion helpers ────────────────────────────────────────────────

def frame_to_gray(pil_frame):
    """Convert a PIL frame to a numpy grayscale array."""
    if pil_frame.mode in ('RGBA', 'LA', 'PA'):
        pil_frame = pil_frame.convert('RGB')
    elif pil_frame.mode != 'RGB':
        pil_frame = pil_frame.convert('RGB')
    return np.array(ImageOps.grayscale(pil_frame))


def frame_to_color_bgr(pil_frame, color_hex, black_hex):
    """Convert a PIL frame to monochrome-tinted BGR array (for opacity blend)."""
    if pil_frame.mode in ('RGBA', 'LA', 'PA'):
        pil_frame = pil_frame.convert('RGB')
    elif pil_frame.mode != 'RGB':
        pil_frame = pil_frame.convert('RGB')
    gray = ImageOps.grayscale(pil_frame)
    mono = ImageOps.colorize(gray, black=black_hex, white=color_hex)
    return np.array(mono.convert('RGB'))[:, :, ::-1].copy()


def cv2_frame_to_gray(bgr_frame):
    """Convert an OpenCV BGR frame to grayscale numpy array."""
    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)


def cv2_frame_to_mono_bgr(bgr_frame, color_hex, black_hex):
    """Convert an OpenCV BGR frame to monochrome-tinted BGR."""
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    # Use PIL for the colorize step
    pil_gray = Image.fromarray(gray)
    mono = ImageOps.colorize(pil_gray, black=black_hex, white=color_hex)
    return np.array(mono.convert('RGB'))[:, :, ::-1].copy()


# ─── Target preparation (still image) ───────────────────────────────────────

def prepare_target(target_path, color_hex, black_hex):
    """
    Load target, convert to monochrome color version (for opacity blend)
    and grayscale (for brightness matching).
    Returns (grayscale_array, color_bgr_array).
    """
    with Image.open(target_path) as img:
        if img.mode in ('RGBA', 'LA', 'PA'):
            img = img.convert('RGB')

        gray = ImageOps.grayscale(img)
        gray_arr = np.array(gray)

        mono = ImageOps.colorize(gray, black=black_hex, white=color_hex)
        mono = mono.convert('RGB')
        color_bgr = np.array(mono)[:, :, ::-1].copy()

    return gray_arr, color_bgr


# ═══════════════════════════════════════════════════════════════════════════════
#  GIF PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_gif(target_path, output_path, tiles, sorted_brightness, sort_order,
                tile_h, tile_w, color_hex, black_hex, best_k, opacity,
                transparent=False):
    """
    Process an animated GIF frame-by-frame, mosaic each frame,
    and reassemble into an output GIF preserving frame timing.
    """
    gif = Image.open(target_path)

    # Extract frame durations
    durations = []
    frames_pil = []
    try:
        while True:
            duration = gif.info.get('duration', 100)  # ms per frame, default 100
            durations.append(duration)
            frames_pil.append(gif.copy())
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass

    n_frames = len(frames_pil)
    print(f"  GIF: {n_frames} frames, size: {frames_pil[0].size}")

    mosaic_frames = []
    for i, frame in enumerate(frames_pil):
        gray = frame_to_gray(frame)
        color_bgr = frame_to_color_bgr(frame, color_hex, black_hex) if opacity > 0 else None

        mosaic_arr = mosaicify(
            gray, tile_h, tile_w,
            tiles, sorted_brightness, sort_order,
            best_k=best_k, opacity=opacity, target_color=color_bgr,
            transparent=transparent,
            verbose=(i == 0),  # only print grid info for first frame
        )

        if transparent:
            # BGRA -> RGBA -> PIL RGBA
            mosaic_rgba = mosaic_arr[:, :, [2, 1, 0, 3]]
            mosaic_pil = Image.fromarray(mosaic_rgba, 'RGBA')
        else:
            # BGR -> RGB -> PIL
            mosaic_rgb = mosaic_arr[:, :, ::-1]
            mosaic_pil = Image.fromarray(mosaic_rgb)
        mosaic_frames.append(mosaic_pil)

        if (i + 1) % 10 == 0 or i == n_frames - 1:
            print(f"  Frame {i + 1}/{n_frames}")

    # Save as animated GIF
    print(f"  Saving GIF ({n_frames} frames)...")

    if transparent:
        # GIF supports only 1-bit transparency via a palette index.
        # Convert each RGBA frame to palette mode with a transparent color.
        palette_frames = []
        for fr in mosaic_frames:
            # Threshold alpha: < 128 becomes fully transparent
            alpha = fr.split()[3]
            # Quantize to 255 colors (reserve index 0 for transparency)
            p_frame = fr.convert('RGB').quantize(colors=255, method=2)
            # Create a mask where alpha < 128
            mask = Image.eval(alpha, lambda a: 255 if a < 128 else 0)
            # Paste transparent index (0) where mask is white
            p_frame.paste(0, mask=mask)
            palette_frames.append(p_frame)

        palette_frames[0].save(
            output_path,
            save_all=True,
            append_images=palette_frames[1:],
            duration=durations,
            loop=0,
            optimize=False,
            transparency=0,
            disposal=2,  # restore to background (transparent) between frames
        )
    else:
        mosaic_frames[0].save(
            output_path,
            save_all=True,
            append_images=mosaic_frames[1:],
            duration=durations,
            loop=0,  # loop forever
            optimize=False,
        )
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  VIDEO PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_video(target_path, output_path, tiles, sorted_brightness, sort_order,
                  tile_h, tile_w, color_hex, black_hex, best_k, opacity, fps=None):
    """
    Process a video file frame-by-frame using OpenCV.
    Writes output as mp4. Does not handle audio (use ffmpeg to merge after).
    """
    cap = cv2.VideoCapture(target_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {target_path}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = fps or src_fps

    print(f"  Video: {src_w}x{src_h}, {total_frames} frames, {src_fps:.1f} fps")

    # Compute output dimensions (trimmed to tile grid)
    n_rows = src_h // tile_h
    n_cols = src_w // tile_w
    out_h = n_rows * tile_h
    out_w = n_cols * tile_w

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h), True)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gray = cv2_frame_to_gray(frame)
        color_bgr = cv2_frame_to_mono_bgr(frame, color_hex, black_hex) if opacity > 0 else None

        mosaic = mosaicify(
            gray, tile_h, tile_w,
            tiles, sorted_brightness, sort_order,
            best_k=best_k, opacity=opacity, target_color=color_bgr,
            verbose=(frame_idx == 0),  # only print grid info for first frame
        )

        writer.write(mosaic)
        frame_idx += 1

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"  Frame {frame_idx}/{total_frames}")

    cap.release()
    writer.release()
    print(f"  Saved: {output_path} ({frame_idx} frames)")

    # Hint about audio
    print(f"\n  NOTE: Audio is not included in the output video.")
    print(f"  To add audio from the original, use ffmpeg:")
    print(f"    ffmpeg -i \"{output_path}\" -i \"{target_path}\" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0? -shortest \"{os.path.splitext(output_path)[0]}_audio.mp4\"")


# ─── Format detection ────────────────────────────────────────────────────────

def detect_format(path):
    """Detect if the target is an image, GIF, or video."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.gif':
        # Check if it's animated
        try:
            img = Image.open(path)
            try:
                img.seek(1)
                img.close()
                return 'gif'
            except EOFError:
                img.close()
                return 'image'  # single-frame GIF
        except Exception:
            return 'image'
    elif ext in ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv'):
        return 'video'
    else:
        return 'image'


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Standalone Blue-In-Blue-Out mosaic generator (no faiss/dlib needed)"
    )

    parser.add_argument("--target", "-t", type=str, required=True,
                        help="Target image, GIF, or video to recreate as a mosaic")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output path (.jpg for images, .gif for GIFs, .mp4 for video)")

    # Tile source (pick one)
    tile_group = parser.add_mutually_exclusive_group()
    tile_group.add_argument("--codebook-dir", dest="codebook_dir", type=str,
                            help="Use existing tile images from this folder")
    tile_group.add_argument("--source-images", dest="source_images", type=str,
                            help="Convert real photos to monochrome tiles")
    tile_group.add_argument("--tile-mode", dest="tile_mode", type=str,
                            choices=["blocks", "halftone", "crosshatch", "noise"],
                            help="Generate synthetic tiles (default: blocks)")

    # Tile settings
    parser.add_argument("--levels", type=int, default=64,
                        help="Number of brightness levels / tiles to generate (default: 64)")
    parser.add_argument("--color", "-c", type=str, default="#0000FF",
                        help="Hex color for bright areas (default: #0000FF blue)")
    parser.add_argument("--black", "-b", dest="black_color", type=str, default="#000000",
                        help="Hex color for dark areas (default: #000000 black)")

    # Mosaic settings
    parser.add_argument("--scale", type=int, default=6,
                        help="Tile scale — smaller = more tiles = finer detail (default: 6)")
    parser.add_argument("--height-aspect", dest="height_aspect", type=float, default=4.0,
                        help="Height aspect (default: 4)")
    parser.add_argument("--width-aspect", dest="width_aspect", type=float, default=3.0,
                        help="Width aspect (default: 3)")
    parser.add_argument("--best-k", dest="best_k", type=int, default=1,
                        help="Pick from top K brightness matches (default: 1)")
    parser.add_argument("--opacity", type=float, default=0.0,
                        help="Blend original on top (0.0=pure mosaic, 0.3=subtle, default: 0.0)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override output FPS for video (default: same as source)")
    parser.add_argument("--transparent", action="store_true",
                        help="Make dark areas transparent instead of black (PNG/GIF only)")

    args = parser.parse_args()

    tile_h = int(args.height_aspect * args.scale)
    tile_w = int(args.width_aspect * args.scale)
    color_rgb = hex_to_rgb(args.color)
    black_rgb = hex_to_rgb(args.black_color)

    # Detect input format
    fmt = detect_format(args.target)

    # ─── Transparent mode validation ──────────────────────────────────────
    if args.transparent:
        # Warn and auto-fix incompatible output formats
        out_ext = os.path.splitext(args.output)[1].lower()
        if fmt == 'video':
            print("  WARNING: --transparent is not supported for video output. Ignoring.")
            args.transparent = False
        elif fmt == 'image' and out_ext in ('.jpg', '.jpeg'):
            new_output = os.path.splitext(args.output)[0] + '.png'
            print(f"  WARNING: JPEG does not support transparency. Switching output to: {new_output}")
            args.output = new_output
        if args.transparent and args.opacity > 0:
            print("  WARNING: --opacity is incompatible with --transparent. Ignoring opacity.")
            args.opacity = 0.0

    print("=" * 60)
    print("  BLUE MOSAIC")
    print("=" * 60)
    print(f"  Target:  {args.target} ({fmt})")
    print(f"  Output:  {args.output}")
    print(f"  Color:   {args.black_color} -> {args.color}")
    print(f"  Scale:   {args.scale} -> tiles are {tile_w}x{tile_h} px")
    print(f"  Levels:  {args.levels} brightness steps")
    if args.transparent:
        print(f"  Mode:    transparent background")
    print()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # ─── Step 1: Get tiles ───────────────────────────────────────────────
    print("Step 1: Preparing tiles...")
    start = time.time()

    if args.codebook_dir:
        tiles = load_tiles_from_dir(args.codebook_dir, tile_h, tile_w)
        if args.transparent:
            tiles = [add_alpha_from_brightness(t, black_rgb) for t in tiles]
    elif args.source_images:
        out_dir = os.path.join(os.path.dirname(args.output), "blue_tiles")
        tiles = convert_photos_to_blue(
            args.source_images, out_dir,
            args.color, args.black_color, tile_h, tile_w)
        if args.transparent:
            tiles = [add_alpha_from_brightness(t, black_rgb) for t in tiles]
    else:
        mode = args.tile_mode or "blocks"
        print(f"  Generating {args.levels} '{mode}' tiles at {tile_w}x{tile_h} px...")
        tiles = TILE_GENERATORS[mode](color_rgb, black_rgb, args.levels, tile_w, tile_h,
                                      transparent=args.transparent)

    if not tiles:
        print("ERROR: No tiles. Exiting.")
        sys.exit(1)

    # Build brightness index (shared across all frames)
    brightnesses = compute_tile_brightness(tiles)
    sort_order, sorted_brightness = build_brightness_index(brightnesses)

    print(f"  {len(tiles)} tiles ready ({time.time() - start:.1f}s)")
    print(f"  Brightness range: {sorted_brightness[0]:.1f} - {sorted_brightness[-1]:.1f}")

    # ─── Step 2: Process based on format ─────────────────────────────────

    if fmt == 'gif':
        print(f"\nStep 2: Processing animated GIF...")
        process_gif(
            args.target, args.output,
            tiles, sorted_brightness, sort_order,
            tile_h, tile_w,
            args.color, args.black_color,
            args.best_k, args.opacity,
            transparent=args.transparent,
        )

    elif fmt == 'video':
        print(f"\nStep 2: Processing video...")
        process_video(
            args.target, args.output,
            tiles, sorted_brightness, sort_order,
            tile_h, tile_w,
            args.color, args.black_color,
            args.best_k, args.opacity,
            fps=args.fps,
        )

    else:
        # Still image
        print(f"\nStep 2: Preparing target image...")
        target_gray, target_color = prepare_target(args.target, args.color, args.black_color)
        print(f"  Target size: {target_gray.shape[1]} x {target_gray.shape[0]} px")

        mono_path = os.path.splitext(args.output)[0] + '_target_mono.jpg'
        cv2.imwrite(mono_path, target_color)
        print(f"  Saved monochrome target: {mono_path}")

        print(f"\nStep 3: Building mosaic...")
        start = time.time()

        mosaic = mosaicify(
            target_gray, tile_h, tile_w,
            tiles, sorted_brightness, sort_order,
            best_k=args.best_k,
            opacity=args.opacity,
            target_color=target_color,
            transparent=args.transparent,
        )

        elapsed = time.time() - start
        print(f"  Built in {elapsed:.1f}s")

        if args.transparent:
            # BGRA -> RGBA for PIL, save as PNG
            mosaic_rgba = mosaic[:, :, [2, 1, 0, 3]]
            Image.fromarray(mosaic_rgba, 'RGBA').save(args.output)
        else:
            cv2.imwrite(args.output, mosaic)
        print(f"\nDone! Saved: {args.output}")
        print(f"  Mosaic size: {mosaic.shape[1]} x {mosaic.shape[0]} px")
        return

    print(f"\nDone! Saved: {args.output}")


if __name__ == '__main__':
    main()
