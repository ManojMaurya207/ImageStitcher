"""
matcher.py – Feature matching between two images.

Uses:
  - BFMatcher with Hamming norm for ORB (binary descriptors)
  - BFMatcher with L2 norm for SIFT (float descriptors)
  - Lowe's ratio test to filter ambiguous matches

Usage:
    from stitching.matcher import match_features
    pts_src, pts_dst = match_features(kp1, des1, kp2, des2, method='ORB')
"""

import cv2
import numpy as np


def match_features(
    kp1: list,
    des1: np.ndarray,
    kp2: list,
    des2: np.ndarray,
    method: str = "ORB",
    ratio_thresh: float = 0.75,
    min_matches: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match descriptors between two images and return corresponding point pairs.

    Parameters
    ----------
    kp1, kp2       : Keypoints from image 1 and image 2.
    des1, des2     : Descriptors from image 1 and image 2.
    method         : 'ORB' or 'SIFT' — determines norm type.
    ratio_thresh   : Lowe's ratio test threshold (default 0.75).
    min_matches    : Minimum good matches required; raises RuntimeError if fewer.

    Returns
    -------
    pts_src : (N, 2) float32 – matched points in image 1.
    pts_dst : (N, 2) float32 – corresponding points in image 2.
    """
    method = method.upper()

    # Choose norm based on descriptor type
    if method == "ORB":
        norm = cv2.NORM_HAMMING
    elif method == "SIFT":
        norm = cv2.NORM_L2
    else:
        raise ValueError(f"Unsupported method '{method}'. Choose 'ORB' or 'SIFT'.")

    bf = cv2.BFMatcher(norm, crossCheck=False)

    # kNN match with k=2 for ratio test
    raw_matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
        elif len(pair) == 1:
            # Fallback for when only 1 neighbor is found
            good.append(pair[0])

    if len(good) < min_matches:
        raise RuntimeError(
            f"Not enough good matches found: {len(good)} (need ≥ {min_matches}). "
            "Try overlapping images more, or switch to SIFT."
        )

    # Extract corresponding point coordinates
    pts_src = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_dst = np.float32([kp2[m.trainIdx].pt for m in good])

    return pts_src, pts_dst


def visualize_matches(
    img1: np.ndarray,
    kp1: list,
    img2: np.ndarray,
    kp2: list,
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
) -> np.ndarray:
    """
    Draw matched keypoints side-by-side for debugging.

    Returns BGR image with matches drawn.
    """
    # Re-create DMatch objects from point index lists
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    for p1, p2 in zip(pts_src, pts_dst):
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]) + w1, int(p2[1])
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(canvas, (x1, y1), 4, (0, 0, 255), -1)
        cv2.circle(canvas, (x2, y2), 4, (255, 0, 0), -1)

    return canvas
