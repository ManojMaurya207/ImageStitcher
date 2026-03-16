"""
src/stitching/__init__.py

Public API for the stitching package.
"""

from .pipeline import stitch_images, stitch_pair
from .feature import detect_and_describe
from .matcher import match_features, visualize_matches
from .homography import compute_homography, warp_and_place
from .blender import blend_images, crop_black_borders, _build_border_mask

__all__ = [
    # High-level API
    "stitch_images",
    "stitch_pair",
    # Low-level helpers (useful for debugging / research)
    "detect_and_describe",
    "match_features",
    "visualize_matches",
    "compute_homography",
    "warp_and_place",
    "blend_images",
    "crop_black_borders",
]
