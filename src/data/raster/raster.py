"""Rasterizer for graph nodes and edges.

Convention: pixel (i, j) covers [i, i+1] x [j, j+1], center (i+0.5, j+0.5).
Canvas: (H, W), float64, 1 = white, 0 = black.

For each shape: compute its coverage (area inside each pixel), then draw it.
"""

import numpy as np
from PIL import ImageColor

# ===== SHARED =====
def _pixel_area(Q, X, Y):
    """Area of the shape inside the pixels whose top-left corners are (X, Y)."""
    return Q(X + 1, Y + 1) - Q(X, Y + 1) - Q(X + 1, Y) + Q(X, Y)



def _draw(canvas, box, coverage, color):
    """Blend `color` into the canvas, weighted by the coverage of each pixel."""
    # avoid drawing outside the canvas
    H, W = canvas.shape[:2]
    x0, x1 = max(int(np.floor(box[0])), 0), min(int(np.ceil(box[1])), W)
    y0, y1 = max(int(np.floor(box[2])), 0), min(int(np.ceil(box[3])), H)
    if x0 >= x1 or y0 >= y1:
        return

    # create pixel coordinates
    Y, X = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    cov = np.clip(coverage(X, Y), 0.0, 1.0)[..., None]      # (h, w, 1)
    region = canvas[y0:y1, x0:x1]                            # (h, w, 3)
    # calculates the area of ​​the shape present within each pixel
    canvas[y0:y1, x0:x1] = region * (1.0 - cov) + to_rgb(color) * cov

def to_rgb(color):
    """Convert 'red', '#ff8800', (255, 0, 0)... to floats in [0, 1]."""
    if isinstance(color, str):
        color = ImageColor.getrgb(color)
    return np.array(color[:3], dtype=np.float64) / 255.0

def to_uint8(canvas):
    """Convert pixel values from [0, 1] to [0, 255]."""
    return (np.clip(canvas, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ===== NODE (disk of center c, radius r) =====

def _disk_quadrant(x, y, r):
    """Calculate the cumulative area of a disk of radius r."""

    # primitive used to calculate the area under the circle's arc (from x**2 + y**2 = r**2)
    G = lambda t: 0.5 * (
        t * np.sqrt(np.maximum(r * r - t * t, 0.0))
        + r * r * np.arcsin(np.clip(t / r, -1.0, 1.0))
    )

    # determine the sign based on the original coordinates, used for symmetry
    s = np.sign(x) * np.sign(y)

    # quarter of the disk (x>0 et y>0)
    x, y = np.minimum(np.abs(x), r), np.minimum(np.abs(y), r)

    # find the horizontal coordinate of the circle's boundary at height y
    # x = sqrt(r**2 - y**2)
    xc = np.sqrt(np.maximum(r * r - y * y, 0.0))

    # If (x, y) is inside the circle, use the rectangle area;
    # otherwise, calculate the area using the circle's arc.
    return s * np.where(
        x * x + y * y <= r * r,  # Is the point inside the circle?
        x * y,                   # Inside -> simple rectangle area
        xc * y + G(x) - G(xc)   # Outside -> area under the circular arc
    )


def node_coverage(X, Y, c, r):
    return _pixel_area(lambda x, y: _disk_quadrant(x - c[0], y - c[1], r), X, Y)

def draw_node(canvas, c, r, color="black"):
    if r <= 0:
        return
    box = (c[0] - r - 1, c[0] + r + 1, c[1] - r - 1, c[1] + r + 1)  # compute only for pixel around the node
    _draw(canvas, box, lambda X, Y: node_coverage(X, Y, c, r), color)

# ===== EDGE (rectangle of width w between a and b, flat caps) =====
def _edge_corners(a, b, w):
    """Rectangle corners in counter-clockwise order."""
    # vector ab
    dx, dy = b[0] - a[0], b[1] - a[1]

    # length ab
    L = np.hypot(dx, dy)

    # vector perpendicular to ab
    nx, ny = -dy / L * w / 2, dx / L * w / 2
    return [(a[0] - nx, a[1] - ny), (b[0] - nx, b[1] - ny), (b[0] + nx, b[1] + ny), (a[0] + nx, a[1] + ny)]


def _polygon_quadrant(x, y, corners):
    """Calculate the cumulative area of a convex polygon."""

    # initialize the total signed area
    total = 0.0

    # iterate over each polygon edge (4)
    for (xa, ya), (xb, yb) in zip(corners, corners[1:] + corners[:1]):
        # skip vertical edges because they have zero horizontal width
        if xa == xb:
            continue

        # determine the sign of this edge's contribution
        sign = -np.sign(xb - xa)

        # order the endpoints from left to right
        (xl, yl), (xr, yr) = sorted([(xa, ya), (xb, yb)])

        # horizontal length included up to x
        l = np.maximum(np.minimum(x, xr) - xl, 0.0) # minimum avoid to go out

        # edge height at the end of this interval
        y2 = yl + (yr - yl) / (xr - xl) * l

        # vertical offsets from the line Y = y
        d1, d2 = yl - y, y2 - y
        dd = d2 - d1

        # correction area above the line Y = y
        above = np.where(
            dd != 0, # the height varies along the segment?
            l * (np.maximum(d2, 0) ** 2 - np.maximum(d1, 0) ** 2) / (2 * np.where(dd != 0, dd, 1.0)), # variation
            l * np.maximum(d1, 0) # constant height
        )

        # add this edge's signed area contribution
        total = total + sign * (l * (yl + y2) / 2 - above)

    return total


def edge_coverage(X, Y, a, b, w):
    corners = _edge_corners(a, b, w)
    return _pixel_area(lambda x, y: _polygon_quadrant(x, y, corners), X, Y)

def draw_edge(canvas, a, b, w, color="black"):
    # check whether the order needs to be draw (w<0 or length too small)
    if w <= 0 or np.hypot(b[0] - a[0], b[1] - a[1]) < 1e-12:
        return

    # marge around the edge
    m = w / 2 + 1
    # calculates the area of ​​the shape present within each pixel
    box = (min(a[0], b[0]) - m, max(a[0], b[0]) + m, min(a[1], b[1]) - m, max(a[1], b[1]) + m)
    _draw(canvas, box, lambda X, Y: edge_coverage(X, Y, a, b, w), color)