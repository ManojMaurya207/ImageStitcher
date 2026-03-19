"""
pipeline.py – End-to-end image stitching pipeline.

Orchestrates: feature detection → matching → homography → warp → blend → crop.

Supports stitching 2 or more images by sequentially merging them left-to-right.

Usage:
    from stitching.pipeline import stitch_images
    panorama = stitch_images([img1, img2, img3], method='ORB')
"""

import cv2
import numpy as np

from .feature import detect_and_describe
from .matcher import match_features
from .homography import compute_homography, warp_and_place
from .blender import blend_images, crop_black_borders


def stitch_pair(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    method: str = "ORB",
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 4.0,
    feather: int = 40,
) -> np.ndarray:
    """
    Stitch a single src_img onto dst_img.

    Parameters
    ----------
    src_img       : Image to warp in.
    dst_img       : Reference / base image.
    method        : 'ORB' (default) or 'SIFT'.
    ratio_thresh  : Lowe's ratio test threshold.
    ransac_thresh : RANSAC reprojection error threshold in pixels.
    feather       : Feather blend width in pixels.

    Returns
    -------
    Merged panorama (not yet cropped).
    """
    print(f"[pipeline] Detecting features (method={method})...")
    
    # --- Avoid OOM by downscaling images just for feature detection ---
    src_scale = min(1.0, 1200.0 / max(src_img.shape[:2]))
    dst_scale = min(src_scale, 3000.0 / max(dst_img.shape[:2]))
    
    src_small = cv2.resize(src_img, (0,0), fx=src_scale, fy=src_scale, interpolation=cv2.INTER_AREA) if src_scale < 1.0 else src_img
    dst_small = cv2.resize(dst_img, (0,0), fx=dst_scale, fy=dst_scale, interpolation=cv2.INTER_AREA) if dst_scale < 1.0 else dst_img

    kp1, des1 = detect_and_describe(src_small, method=method)
    kp2, des2 = detect_and_describe(dst_small, method=method)
    print(f"[pipeline] Found {len(kp1)} / {len(kp2)} keypoints in src / dst.")

    print("[pipeline] Matching features...")
    pts_src, pts_dst = match_features(kp1, des1, kp2, des2, method=method, ratio_thresh=ratio_thresh)
    print(f"[pipeline] {len(pts_src)} good matches after ratio test.")

    # Scale coordinates back up to original image resolution
    pts_src = pts_src / src_scale
    pts_dst = pts_dst / dst_scale

    print("[pipeline] Computing homography (RANSAC)...")
    H, rmse = compute_homography(pts_src, pts_dst, ransac_thresh=ransac_thresh)

    print("[pipeline] Warping and placing images on canvas...")
    warped_src, offset = warp_and_place(src_img, dst_img, H)

    print("[pipeline] Blending seam...")
    blended = blend_images(warped_src, dst_img, offset, feather=feather)

    return blended


def stitch_images(
    images: list[np.ndarray],
    method: str = "ORB",
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 4.0,
    feather: int = 40,
    crop: bool = True,
) -> np.ndarray:
    """
    Stitch a list of 2+ images into a single panorama.

    Images are merged sequentially: images[0] is stitched onto images[1],
    then the result is stitched with images[2], etc.

    Parameters
    ----------
    images        : List of BGR numpy arrays (at least 2).
    method        : Feature detector: 'ORB' or 'SIFT'.
    ratio_thresh  : Lowe's ratio test threshold (default 0.75).
    ransac_thresh : RANSAC inlier threshold in pixels (default 4.0).
    feather       : Blend ramp width in pixels (default 40).
    crop          : Whether to crop black borders after stitching (default True).

    Returns
    -------
    panorama : Final stitched (and optionally cropped) BGR image.

    Raises
    ------
    ValueError   : If fewer than 2 images are supplied.
    RuntimeError : If feature matching or homography estimation fails.
    """
    if len(images) < 2:
        raise ValueError(f"At least 2 images are required, got {len(images)}.")

    for i, img in enumerate(images):
        if img is None or img.size == 0:
            raise ValueError(f"Image at index {i} is None or empty.")

    print(f"[pipeline] Starting stitching of {len(images)} image(s)...")

    # Start with the first two images, warping images[1] onto images[0] (base)
    try:
        panorama = stitch_pair(
            images[1], images[0],
            method=method,
            ratio_thresh=ratio_thresh,
            ransac_thresh=ransac_thresh,
            feather=feather,
        )
    except RuntimeError as e:
        print(f"[pipeline] Warning: Failed to stitch initial pair. Error: {e}")
        print("[pipeline] Falling back to the first image.")
        panorama = images[0]

    # Sequentially add remaining images
    for i, next_img in enumerate(images[2:], start=2):
        print(f"[pipeline] Adding image {i + 1}/{len(images)}...")
        try:
            # Warp next_img onto the evolving panorama
            panorama = stitch_pair(
                next_img, panorama,
                method=method,
                ratio_thresh=ratio_thresh,
                ransac_thresh=ransac_thresh,
                feather=feather,
            )
        except RuntimeError as e:
            print(f"[pipeline] Warning: Failed to stitch image {i + 1}. Skipping. Error: {e}")
            continue

    if crop:
        print("[pipeline] Cropping black borders...")
        panorama = crop_black_borders(panorama)

    print("[pipeline] Done.")
    return panorama
