"""
blender.py – Image blending and smart crop.

Two main responsibilities:
1. blend_images(warped, base_img, offset)
     Feather-blends the overlap region between the warped panorama canvas
     and the base/destination image so seams are less visible.

2. crop_black_borders(image)
     Crops the black border padding left by warping using the
     Maximal-Inscribed-Rectangle algorithm (largest rectangle inside
     the valid-pixel mask), ported from tests_space/AlgoForCropAnime.py.

Usage:
    from stitching.blender import blend_images, crop_black_borders
    blended = blend_images(canvas, dst_img, offset)
    result  = crop_black_borders(blended)
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------

def blend_images(
    canvas: np.ndarray,
    base_img: np.ndarray,
    offset: tuple[int, int],
    feather: int = 40,
) -> np.ndarray:
    """
    Alpha-feather blend the overlap region between the warped canvas and
    the base image.

    Parameters
    ----------
    canvas   : Full panorama canvas (output of warp_and_place).
    base_img : The reference image placed on the canvas.
    offset   : (tx, ty) shift used when placing base_img on canvas.
    feather  : Width of the blending gradient in pixels (default 40).

    Returns
    -------
    Blended BGR image (same size as canvas).
    """
    tx, ty = offset
    if tx < 0 or ty < 0:
        raise ValueError(f"Offset cannot be negative: {offset}")

    h_b, w_b = base_img.shape[:2]
    result = canvas.copy().astype(np.float32)

    # Region on canvas where base_img sits
    y1, y2 = ty, ty + h_b
    x1, x2 = tx, tx + w_b

    # Clip to canvas bounds
    y2 = min(y2, canvas.shape[0])
    x2 = min(x2, canvas.shape[1])

    base_region = base_img[:y2 - y1, :x2 - x1].astype(np.float32)
    canvas_region = canvas[y1:y2, x1:x2].astype(np.float32)

    # Build a horizontal feather mask: blend warped ← left seam region → base
    # The mask is 1 for full base, 0 for full warped
    mask = np.ones((y2 - y1, x2 - x1), dtype=np.float32)

    # Identify columns where *both* canvas and base are non-zero (overlap)
    canvas_gray = cv2.cvtColor(canvas[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    base_gray = cv2.cvtColor(base_img[:y2 - y1, :x2 - x1], cv2.COLOR_BGR2GRAY)
    overlap_cols = np.where((canvas_gray.mean(axis=0) > 0) & (base_gray.mean(axis=0) > 0))[0]

    if len(overlap_cols) > 0:
        ol_start = overlap_cols[0]
        ol_end = overlap_cols[-1]
        ol_width = max(ol_end - ol_start + 1, 1)
        ramp = np.linspace(0, 1, ol_width)
        # Apply blend over the full overlap width
        mask[:, ol_start:ol_start + ol_width] = ramp
        mask[:, :ol_start] = 0  # pure warped side

    mask_3ch = np.stack([mask] * 3, axis=2)
    blended_region = canvas_region * (1 - mask_3ch) + base_region * mask_3ch
    result[y1:y2, x1:x2] = blended_region

    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# Crop – Maximal Inscribed Rectangle
# ---------------------------------------------------------------------------

def _maximal_inscribed_rectangle(binary_mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Find the largest axis-aligned rectangle that fits entirely inside the
    binary mask (1 = valid pixel, 0 = black/invalid).

    Uses the largest-rectangle-in-histogram trick row by row.

    Returns
    -------
    (x, y, width, height) of the maximal rectangle.
    """
    h, w = binary_mask.shape
    heights = [0] * w
    max_area = 0
    max_rect = (0, 0, 0, 0)

    for row in range(h):
        # Update column heights
        for col in range(w):
            heights[col] = heights[col] + 1 if binary_mask[row, col] == 1 else 0

        # Largest rectangle in histogram (stack-based O(w))
        stack: list[int] = []
        for i in range(w + 1):
            cur_h = heights[i] if i < w else 0
            while stack and cur_h < heights[stack[-1]]:
                rect_h = heights[stack.pop()]
                left = stack[-1] + 1 if stack else 0
                rect_w = i - left
                area = rect_h * rect_w
                if area > max_area:
                    max_area = area
                    max_rect = (left, row - rect_h + 1, rect_w, rect_h)
            stack.append(i)

    return max_rect


def _build_border_mask(image: np.ndarray) -> np.ndarray:
    """
    Build a binary mask that marks the TRUE valid canvas region.

    The naive approach (threshold gray > 1) fails when the image has dark or
    black content in the interior (e.g. a dark seam, shadow, or black object)
    because those interior dark pixels get treated as "invalid border".

    Instead we flood-fill outward from all four corners to identify the
    warped-perspective background (the actual black border left by cv2.warpPerspective).
    Pixels reached by the flood fill are background; everything else is valid content
    — even if it is dark or fully black.

    Returns
    -------
    binary : uint8 mask, 1 = valid content, 0 = warped-border background.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Initial coarse mask: 0 where pixel is black (unused canvas)
    _, raw = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # We flood-fill from every edge pixel that is black into a copy of raw.
    # floodFill replaces connected 0-regions reachable from the seed with 128
    # (a sentinel value). Regions NOT reached are interior – valid content.
    filled = raw.copy()
    flood_flags = 4 | (128 << 8) | cv2.FLOODFILL_MASK_ONLY

    # Use a (h+2) x (w+2) mask as required by floodFill
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Seed from all four border edges (only black pixels are flood-fill seeds)
    seeds = set()
    for x in range(w):
        if raw[0, x] == 0:     seeds.add((x, 0))
        if raw[h-1, x] == 0:   seeds.add((x, h - 1))
    for y in range(h):
        if raw[y, 0] == 0:     seeds.add((0, y))
        if raw[y, w-1] == 0:   seeds.add((w - 1, y))

    for (sx, sy) in seeds:
        if ff_mask[sy + 1, sx + 1] == 0:   # not yet visited
            cv2.floodFill(filled, ff_mask, (sx, sy), 128,
                          loDiff=0, upDiff=0, flags=flood_flags)

    # ff_mask==1 marks pixels that were flood-filled (border background)
    border_bg = (ff_mask[1:-1, 1:-1] == 1).astype(np.uint8)

    # Valid = not border background AND is on the canvas (has some content)
    # We also keep interior dark pixels as valid by NOT using raw here.
    # Instead: valid = everything that is NOT tagged border background.
    valid = (1 - border_bg).astype(np.uint8)

    # Small morphological closing fills any tiny isolated holes from jpeg
    # compression artefacts at the very edge, preventing height resets in MIR.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel)

    return valid


def crop_black_borders(image: np.ndarray) -> np.ndarray:
    """
    Remove black/zero-padding borders introduced by perspective warping.

    Uses the Maximal-Inscribed-Rectangle algorithm on the *border-aware* mask
    (see _build_border_mask) so that dark image content in the panorama interior
    is never mistaken for an invalid border, preventing incorrect top-edge crops.

    Parameters
    ----------
    image : BGR panorama with black borders.

    Returns
    -------
    Cropped BGR image with no black padding.
    """
    binary = _build_border_mask(image)

    x, y, w, h = _maximal_inscribed_rectangle(binary)

    if w == 0 or h == 0:
        # Fallback: bounding box of the largest valid contour
        contours, _ = cv2.findContours(
            binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            xb, yb, wb, hb = cv2.boundingRect(max(contours, key=cv2.contourArea))
            return image[yb:yb + hb, xb:xb + wb]
        return image

    return image[y:y + h, x:x + w]
