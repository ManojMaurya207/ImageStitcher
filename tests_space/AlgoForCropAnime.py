import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------------
# Maximal-inscribed-rectangle + snapshot capture
# ---------------------------------------------------------
def maximal_inscribed_rectangle(binary_mask):
    h, w = binary_mask.shape
    heights = [0] * w
    max_area = 0
    max_rect = (0, 0, 0, 0)

    snapshots = []

    for row in range(h):
        # --- update heights ---
        for col in range(w):
            heights[col] = heights[col] + 1 if binary_mask[row, col] == 1 else 0

        snapshots.append({
            "binary": binary_mask.copy(),
            "row": row,
            "heights": heights.copy(),
            "rect": max_rect,
            "event": "Updating heights"
        })

        # --- largest-rectangle-in-histogram ---
        stack = []
        for i in range(w + 1):
            cur_height = heights[i] if i < w else 0

            while stack and cur_height < heights[stack[-1]]:
                h_rect = heights[stack.pop()]
                left = stack[-1] + 1 if stack else 0
                width = i - left
                area = h_rect * width

                if area > max_area:
                    max_area = area
                    x = left
                    y = row - h_rect + 1
                    max_rect = (x, y, width, h_rect)

                    snapshots.append({
                        "binary": binary_mask.copy(),
                        "row": row,
                        "heights": heights.copy(),
                        "rect": max_rect,
                        "event": "New maximal rectangle"
                    })

            stack.append(i)

    return max_rect, snapshots


# ---------------------------------------------------------
# Sample Input
# ---------------------------------------------------------
binary_mask = np.array([
    [1,1,1,1,0],
    [1,1,1,1,0],
    [1,1,1,1,1],
    [0,1,1,1,1],
], dtype=int)

final_rect, snapshots = maximal_inscribed_rectangle(binary_mask)
fx, fy, fw, fh = final_rect

# Print final bounding box
print("Final Maximal Rectangle:")
print(f"  x={fx}, y={fy}, width={fw}, height={fh}")


# ---------------------------------------------------------
# Animation
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))

def draw_frame(k):
    ax.clear()
    snap = snapshots[k]
    bm = snap["binary"]
    heights = snap["heights"]
    x, y, w, h = snap["rect"]

    # Show mask as a grid-aligned image
    ax.imshow(bm, cmap="Greys", vmin=0, vmax=1, extent=[0, bm.shape[1], bm.shape[0], 0])

    ax.set_title(f"Step {k+1}/{len(snapshots)} – {snap['event']}", fontsize=15)

    # Highlight current row (full cell boundary, not center)
    ax.add_patch(plt.Rectangle(
        (0, snap["row"]),  # top-left corner
        bm.shape[1],       # width
        1,                 # height = 1 cell
        fill=False,
        edgecolor="yellow",
        linewidth=2,
    ))

    # Draw histogram bars aligned to the grid
    for col, hval in enumerate(heights):
        if hval > 0:
            ax.add_patch(plt.Rectangle(
                (col, snap["row"] - hval + 1),  # top-left corner
                1,          # width = 1 cell
                hval,       # height = stack height
                fill=True,
                alpha=0.25,
                color="cyan"
            ))

            ax.text(col + 0.5, snap["row"] - hval + 1 + 0.5,
                    str(hval), color="blue", ha="center", va="center", fontsize=10)

    # Draw current best rectangle
    if w > 0 and h > 0:
        ax.add_patch(plt.Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            edgecolor="red",
            linewidth=3,
        ))

    # Force grid cell lines + integer ticks
    ax.set_xticks(np.arange(0, bm.shape[1] + 1, 1))
    ax.set_yticks(np.arange(0, bm.shape[0] + 1, 1))

    ax.grid(color="black", linewidth=1, linestyle="-", alpha=0.3)
    ax.set_xlim(0, bm.shape[1])
    ax.set_ylim(bm.shape[0], 0)  # invert Y for image-like coordinates

ani = FuncAnimation(fig, draw_frame, frames=len(snapshots), interval=700)
plt.show()
