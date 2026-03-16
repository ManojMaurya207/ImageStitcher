"""
homography.py – Homography estimation and image warping.

Usage:
    from stitching.homography import compute_homography, warp_and_place
    H = compute_homography(pts_src, pts_dst)
    canvas = warp_and_place(src_img, dst_img, H)
"""

import cv2
import numpy as np


def compute_homography(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    ransac_thresh: float = 4.0,
) -> np.ndarray:
    """
    Estimate a 3×3 homography matrix mapping pts_src → pts_dst using RANSAC.

    Parameters
    ----------
    pts_src       : (N, 2) float32 – source points (in image being warped).
    pts_dst       : (N, 2) float32 – destination points (in reference image).
    ransac_thresh : Max pixel error to consider a match an inlier (default 4.0).

    Returns
    -------
    H : (3, 3) float64 homography matrix.

    Raises
    ------
    RuntimeError if fewer than 4 point pairs are provided or RANSAC fails.
    """
    if len(pts_src) < 4:
        raise RuntimeError(
            f"At least 4 matches are required for homography estimation, got {len(pts_src)}."
        )

    H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, ransac_thresh)

    if H is None:
        raise RuntimeError(
            "RANSAC homography estimation failed. "
            "Try increasing overlap between images or using SIFT."
        )

    n_inliers = int(mask.sum()) if mask is not None else 0
    print(f"[homography] H estimated — {n_inliers}/{len(pts_src)} inliers (RANSAC thresh={ransac_thresh}px)")

    return H


def warp_and_place(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    Warp src_img onto a canvas large enough to hold both src_img and dst_img,
    then composite dst_img on top (overwriting the overlap region for now;
    blending is handled in blender.py).

    The canvas size is computed by projecting all four corners of src_img through H
    and finding the union bounding box.

    Parameters
    ----------
    src_img : Image to warp (the one being stitched in).
    dst_img : Reference / base image.
    H       : Homography that maps src_img → dst_img coordinate space.

    Returns
    -------
    canvas : BGR image containing both images placed in their correct positions.
             Also returns the translation offset used.
    offset : (tx, ty) integer offset applied so that warped src_img stays in frame.
    """
    h_src, w_src = src_img.shape[:2]
    h_dst, w_dst = dst_img.shape[:2]

    # Project the four corners of src_img through H
    corners_src = np.float32([
        [0, 0], [w_src, 0], [w_src, h_src], [0, h_src]
    ]).reshape(-1, 1, 2)
    corners_warped = cv2.perspectiveTransform(corners_src, H)

    # Also include the corners of dst_img
    corners_dst = np.float32([
        [0, 0], [w_dst, 0], [w_dst, h_dst], [0, h_dst]
    ]).reshape(-1, 1, 2)

    all_corners = np.concatenate([corners_warped, corners_dst], axis=0)

    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    # Translation offset to shift everything into positive coordinates
    tx = -x_min if x_min < 0 else 0
    ty = -y_min if y_min < 0 else 0
    offset = (tx, ty)

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    if canvas_w <= 0 or canvas_h <= 0:
        raise RuntimeError("Degenerate homography resulted in invalid canvas dimensions. Check overlap.")

    # Build translation matrix
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0,  1]], dtype=np.float64)

    # Warp src_img onto the canvas
    H_shifted = T @ H
    canvas = cv2.warpPerspective(src_img, H_shifted, (canvas_w, canvas_h))

    # Place dst_img on the canvas (non-zero pixels of dst overwrite canvas)
    paste_w = min(w_dst, canvas_w - tx)
    paste_h = min(h_dst, canvas_h - ty)
    if paste_w > 0 and paste_h > 0:
        canvas[ty:ty + paste_h, tx:tx + paste_w] = dst_img[:paste_h, :paste_w]

    return canvas, offset
