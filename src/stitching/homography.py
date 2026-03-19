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
) -> tuple[np.ndarray, float]:
    """
    Estimate a 3x3 homography matrix mapping pts_src -> pts_dst using RANSAC.

    Parameters
    ----------
    pts_src       : (N, 2) float32 - source points (in image being warped).
    pts_dst       : (N, 2) float32 - destination points (in reference image).
    ransac_thresh : Max pixel error to consider a match an inlier (default 4.0).

    Returns
    -------
    H    : (3, 3) float64 homography matrix.
    rmse : float, root mean square alignment error of the inliers.

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
    
    # Calculate RMSE over inliers
    inliers_src = pts_src[(mask.ravel() == 1)]
    inliers_dst = pts_dst[(mask.ravel() == 1)]
    
    if len(inliers_src) > 0:
        inliers_src_homo = np.concatenate([inliers_src, np.ones((len(inliers_src), 1))], axis=1)
        projected = (H @ inliers_src_homo.T).T
        projected = projected[:, :2] / projected[:, 2:]
        rmse = float(np.sqrt(np.mean(np.sum((projected - inliers_dst)**2, axis=1))))
    else:
        rmse = float('inf')

    print(f"[homography] H estimated — {n_inliers}/{len(pts_src)} inliers, RMSE={rmse:.2f}px")

    return H, rmse


def warp_and_place(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    Warp src_img onto a canvas large enough to hold both src_img and dst_img.
    Unlike the previous naive placement, this function no longer overwrites
    the overlap with dst_img. Proper compositing is left entirely to blender.py.

    Parameters
    ----------
    src_img : Image to warp (the one being stitched in).
    dst_img : Reference / base image.
    H       : Homography that maps src_img → dst_img coordinate space.

    Returns
    -------
    warped_src : BGR image containing the warped src_img on the expanded canvas.
    offset     : (tx, ty) integer offset so blender knows where to place dst_img.
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
        
    if canvas_w > 20000 or canvas_h > 20000:
        raise RuntimeError(f"Canvas layout excessively stretched ({canvas_w}x{canvas_h}). Unstable homography.")

    # Build translation matrix
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0,  1]], dtype=np.float64)

    # Warp src_img onto the canvas using high-quality interpolation
    H_shifted = T @ H
    canvas = cv2.warpPerspective(src_img, H_shifted, (canvas_w, canvas_h), flags=cv2.INTER_LANCZOS4)

    return canvas, offset
