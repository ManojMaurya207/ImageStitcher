"""
main.py – CLI entry point for the Image Stitching pipeline.

Usage examples:
    # Stitch all JPEGs in data/sample/ using ORB (default)
    python main.py

    # Specify a folder explicitly
    python main.py --input data/sample --output output/result.jpg

    # Use SIFT and display the result
    python main.py --input data/sample --output output/pano.jpg --method SIFT --show

    # Pass individual files
    python main.py --input data/sample/1.jpg data/sample/2.jpg data/sample/3.jpg
"""

import argparse
import glob
import os
import sys

import cv2

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from stitching.pipeline import stitch_images


def parse_args():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Image Stitcher – build panoramas from overlapping photos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input", "-i",
        nargs="+",
        default=["data/sample"],
        metavar="PATH",
        help=(
            "One or more image files, OR a single directory/glob pattern. "
            "Supported extensions: jpg, jpeg, png, bmp. "
            "Default: data/sample"
        ),
    )

    parser.add_argument(
        "--output", "-o",
        default="output/result.jpg",
        metavar="PATH",
        help="Output file path (default: output/result.jpg).",
    )

    parser.add_argument(
        "--method", "-m",
        default="ORB",
        choices=["ORB", "SIFT"],
        help="Feature detector to use (default: ORB).",
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe's ratio test threshold (default: 0.75).",
    )

    parser.add_argument(
        "--ransac",
        type=float,
        default=4.0,
        help="RANSAC reprojection error threshold in pixels (default: 4.0).",
    )

    parser.add_argument(
        "--feather",
        type=int,
        default=40,
        help="Feather blend width in pixels (default: 40).",
    )

    parser.add_argument(
        "--no-crop",
        action="store_true",
        default=False,
        help="Disable automatic black-border cropping.",
    )

    parser.add_argument(
        "--show", "-s",
        action="store_true",
        default=False,
        help="Display the result in an OpenCV window.",
    )

    return parser.parse_args()


def resolve_images(input_paths: list[str]) -> list[str]:
    """
    Resolve a list of file paths, directories, or glob patterns into a
    sorted list of image file paths.
    """
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    found = []

    for p in input_paths:
        if os.path.isdir(p):
            # Gather all supported images from directory
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.JPG", "*.JPEG", "*.PNG", "*.TIFF", "*.TIF"):
                found.extend(glob.glob(os.path.join(p, ext)))
        elif any(c in p for c in ("*", "?")):
            found.extend(glob.glob(p))
        elif os.path.isfile(p):
            found.append(p)
        else:
            print(f"[warning] Path not found: '{p}' — skipping.")

    # Sort for consistent ordering, deduplicate
    found = sorted(set(found))

    # Filter by extension
    found = [f for f in found if os.path.splitext(f)[1].lower() in supported_exts]

    return found


def main():
    args = parse_args()

    if not (0.0 < args.ratio <= 1.0):
        print("[error] --ratio must be between 0.0 and 1.0")
        sys.exit(1)
    if args.ransac <= 0.0:
        print("[error] --ransac must be greater than 0.0")
        sys.exit(1)

    # ── Resolve input images ─────────────────────────────────────────────────
    image_paths = resolve_images(args.input)

    if len(image_paths) == 0:
        print("[error] No images found. Check --input path.")
        sys.exit(1)

    if len(image_paths) < 2:
        print(f"[error] Need at least 2 images to stitch, found: {len(image_paths)}.")
        sys.exit(1)

    print(f"[main] Found {len(image_paths)} image(s):")
    for p in image_paths:
        print(f"       {p}")

    # ── Load images ──────────────────────────────────────────────────────────
    images = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[warning] Could not read '{p}' — skipping.")
            continue
        images.append(img)

    if len(images) < 2:
        print("[error] Could not load at least 2 valid images.")
        sys.exit(1)

    # ── Stitch ───────────────────────────────────────────────────────────────
    try:
        panorama = stitch_images(
            images,
            method=args.method,
            ratio_thresh=args.ratio,
            ransac_thresh=args.ransac,
            feather=args.feather,
            crop=not args.no_crop,
        )
    except RuntimeError as e:
        print(f"[error] Stitching failed: {e}")
        sys.exit(1)

    # ── Save output ──────────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    success = cv2.imwrite(args.output, panorama)
    if success:
        print(f"[main] Panorama saved → {args.output}  ({panorama.shape[1]}×{panorama.shape[0]} px)")
    else:
        print(f"[error] Failed to write output to '{args.output}'.")

    # ── Optional display ─────────────────────────────────────────────────────
    if args.show:
        # Resize for display if very large
        max_disp_w = 1400
        if panorama.shape[1] > max_disp_w:
            scale = max_disp_w / panorama.shape[1]
            disp = cv2.resize(panorama, None, fx=scale, fy=scale)
        else:
            disp = panorama

        cv2.imshow("Panorama – press any key to close", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()