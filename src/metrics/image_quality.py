"""Quality measures computed on the RENDERED image, at the resolution it is displayed at.

The measures of evaluation.py (and of the literature) are geometric: a node is a point, an edge a segment
without thickness. They cannot see what happens once the drawing is rasterized: two disks 0.3 px apart do
not overlap, yet they merge into one blob of ink; an edge passing 0.5 px from a node seems to touch it. These
measures are computed on the exact pixel coverage of every node and edge (area of the shape inside each
pixel, data/raster/raster.py), for a given image size, node radius and edge width.

For evaluation only (numpy, not differentiable). Conventions: Layout's (an integer coordinate is the center
of a pixel, y pointing down); the frame covers [-0.5, W - 0.5] x [-0.5, H - 0.5].

Each element (node or edge) is stored sparsely: the flat indices y * W + x of the pixels it touches inside
the frame, and its coverage of each of them. A drawing then costs about sum(disk areas) + sum(edge lengths x
(edge width + 3)) values instead of one W x H image per element, which keeps a sweep over thousands of
drawings tractable. Curved edges (src/curves) plug in through the `edges` argument of image_quality.

Measures (image_quality returns them in a dict):
- node_visibility: mean over the nodes of the share of each disk that is visible, i.e. inside the frame
  and not covered by another node; hidden_nodes: share of the nodes less than half visible.
- merged_nodes: nodes that touch another node once rasterized: number of nodes minus number of connected
  blobs of node ink (pixels at least half inked, 8-connectivity), divided by the number of nodes.
- ambiguous_edges: pairs (edge, node that is not one of its ends) sharing a pixel (both coverages above
  `threshold`), divided by the number of edges: an edge that seems connected to a node.
- edge_overlap: share of the edge ink, outside the nodes, covered by two edges or more.
- ink_density: share of the frame covered by ink; ink_peak: densest 16 x 16 window.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from data.raster import raster

EPS = 1e-12


@dataclass
class Element:
    """Sparse coverage of one shape: pixels (flat indices y * W + x, unique, inside the frame) and their
    coverage in (0, 1]."""
    pixels: np.ndarray  # (K,) int64
    coverage: np.ndarray  # (K,) float64


# -----------------------------------------------------------------------------------------------------------
# Coverage of each element
# -----------------------------------------------------------------------------------------------------------
def _sparse(X, Y, cov, width, height):
    """Keep the pixels inside the frame with a nonzero coverage. X, Y: top-left corners (raster convention)."""
    X, Y, cov = X.ravel(), Y.ravel(), np.clip(cov.ravel(), 0.0, 1.0)
    keep = (X >= 0) & (X < width) & (Y >= 0) & (Y < height) & (cov > EPS)
    return Element((Y[keep] * width + X[keep]).astype(np.int64), cov[keep])


def node_element(center, radius, width, height):
    """Exact coverage of a disk; center in Layout coordinates."""
    c = (center[0] + 0.5, center[1] + 0.5)  # raster convention: pixel i spans [i, i + 1]
    Y, X = np.mgrid[math.floor(c[1] - radius) - 1: math.ceil(c[1] + radius) + 1,
                    math.floor(c[0] - radius) - 1: math.ceil(c[0] + radius) + 1]
    return _sparse(X, Y, raster.node_coverage(X.astype(float), Y.astype(float), c, radius), width, height)


def band_pixels(points, reach, width, height):
    """Top-left corners (X, Y) of every pixel within `reach` of the polyline through `points` (M, 2), in
    raster coordinates: candidates for the coverage of a stroke of half-width < reach - 1.5.

    The polyline is sampled every 0.5 px, and every pixel within a square of half-side ceil(reach) around a
    sample is kept: a pixel that meets the stroke has a corner within half-width + sqrt(2) of the polyline,
    hence within 0.25 + half-width + sqrt(2) < reach of a sample (checked against a full grid in the tests).
    """
    points = np.asarray(points, dtype=np.float64)
    samples = [points[:1]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(int(math.ceil(np.hypot(*(b - a)) / 0.5)), 1)
        t = np.arange(1, n + 1)[:, None] / n
        samples.append(a + t * (b - a))
    samples = np.floor(np.concatenate(samples)).astype(np.int64)
    k = int(math.ceil(reach))
    offsets = np.arange(-k, k + 1)
    X = (samples[:, 0, None, None] + offsets[None, None, :]).repeat(len(offsets), axis=1)
    Y = (samples[:, 1, None, None] + offsets[None, :, None]).repeat(len(offsets), axis=2)
    keep = (X >= 0) & (X < width) & (Y >= 0) & (Y < height)
    flat = np.unique(Y[keep] * width + X[keep])
    return flat % width, flat // width


def edge_element(a, b, edge_width, width, height):
    """Exact coverage of a straight edge (rectangle with flat caps, as drawn by data/output.save_png)."""
    a, b = (a[0] + 0.5, a[1] + 0.5), (b[0] + 0.5, b[1] + 0.5)
    if edge_width <= 0 or math.hypot(b[0] - a[0], b[1] - a[1]) < 1e-12:
        return Element(np.empty(0, np.int64), np.empty(0))
    X, Y = band_pixels(np.array([a, b]), edge_width / 2 + 2.0, width, height)
    cov = raster.edge_coverage(X.astype(float), Y.astype(float), a, b, edge_width)
    return _sparse(X, Y, cov, width, height)


def elements(positions, graph, width, height):
    """Node and edge elements of a straight-line drawing. positions: (N, 2) array or tensor."""
    P = _numpy(positions)
    radius = _numpy(graph.node_size).reshape(-1)
    w = float(graph.edge_width)
    nodes = [node_element(P[k], float(radius[k]), width, height) for k in range(len(P))]
    edges = [edge_element(P[u], P[v], w, width, height) for u, v in _numpy(graph.edge_index).T.astype(int)]
    return nodes, edges


def _numpy(t):
    return t.detach().cpu().double().numpy() if hasattr(t, "detach") else np.asarray(t, dtype=np.float64)


# -----------------------------------------------------------------------------------------------------------
# Measures
# -----------------------------------------------------------------------------------------------------------
def _stack(elems):
    """All elements in two flat arrays, with the owner of each value."""
    if not elems:
        return np.empty(0, np.int64), np.empty(0), np.empty(0, np.int64)
    pixels = np.concatenate([e.pixels for e in elems])
    coverage = np.concatenate([e.coverage for e in elems])
    owner = np.repeat(np.arange(len(elems)), [len(e.pixels) for e in elems])
    return pixels, coverage, owner


def _ink(pixels, coverage, size):
    """Union of the coverages, pixel by pixel: 1 - prod(1 - c) (the elements as independent masks)."""
    log_clear = np.zeros(size)
    np.add.at(log_clear, pixels, np.log1p(-np.minimum(coverage, 1 - 1e-9)))
    return -np.expm1(log_clear), log_clear


def _pairs_sharing_pixels(pix_a, own_a, pix_b, own_b):
    """Unique pairs (owner a, owner b) that have at least one pixel in common."""
    order = np.argsort(pix_b, kind="stable")
    pix_b, own_b = pix_b[order], own_b[order]
    lo, hi = np.searchsorted(pix_b, pix_a, "left"), np.searchsorted(pix_b, pix_a, "right")
    count = hi - lo
    a = np.repeat(own_a, count)
    start = np.repeat(lo - np.cumsum(count) + count, count)  # index in pix_b of each repeated match
    b = own_b[start + np.arange(count.sum())]
    return np.unique(np.stack((a, b), axis=1), axis=0) if len(a) else np.empty((0, 2), np.int64)


def image_quality(positions, graph, width, height, edges=None, threshold=0.1, window=16):
    """All the measures of the module docstring, for one drawing.

    positions: (N, 2) Layout positions; graph: a GraphInput (node_size, edge_width, edge_index).
    edges: list of Element, one per edge of graph.edge_index (e.g. curved edges); straight edges if None.
    threshold: coverage above which a pixel counts as touched by an element (ambiguity, overlaps).
    """
    size = width * height
    nodes, straight = elements(positions, graph, width, height)
    edges = straight if edges is None else edges
    radius = _numpy(graph.node_size).reshape(-1)
    n, m = len(nodes), len(edges)
    n_pix, n_cov, n_own = _stack(nodes)
    e_pix, e_cov, e_own = _stack(edges)

    # visibility: part of each disk inside the frame and not covered by the OTHER nodes
    node_ink, log_clear = _ink(n_pix, n_cov, size)
    own_clear = np.log1p(-np.minimum(n_cov, 1 - 1e-9))
    others_clear = np.exp(log_clear[n_pix] - own_clear)  # prod over j != k of (1 - c_j)
    visible = np.bincount(n_own, weights=n_cov * others_clear, minlength=n) / (math.pi * radius ** 2)
    visible = np.clip(visible, 0.0, 1.0)

    # merged nodes: connected blobs of node ink; each node belongs to the blob of its most covered pixel
    labels, _ = ndimage.label((node_ink >= 0.5).reshape(height, width), structure=np.ones((3, 3)))
    labels = labels.reshape(-1)
    node_blob = np.zeros(n, np.int64)
    for k, e in enumerate(nodes):
        if len(e.pixels):
            node_blob[k] = labels[e.pixels[np.argmax(e.coverage)]]
    seen = node_blob[node_blob > 0]
    merged = len(seen) - len(np.unique(seen))

    # ambiguous edges: an edge and a node that is not one of its ends share a pixel
    ends = _numpy(graph.edge_index).T.astype(int)
    tn, te = n_cov > threshold, e_cov > threshold
    pairs = _pairs_sharing_pixels(e_pix[te], e_own[te], n_pix[tn], n_own[tn])
    if len(pairs):
        incident = (ends[pairs[:, 0], 0] == pairs[:, 1]) | (ends[pairs[:, 0], 1] == pairs[:, 1])
        pairs = pairs[~incident]

    # edge overlap: edge ink outside the nodes, covered by two edges or more
    edge_count = np.bincount(e_pix[te], minlength=size)
    edge_ink = (edge_count > 0) & (node_ink <= threshold)
    edge_overlap = float(((edge_count >= 2) & edge_ink).sum() / max(edge_ink.sum(), 1))

    # clutter: union of all the ink, globally and in the densest window
    all_ink, _ = _ink(np.concatenate((n_pix, e_pix)), np.concatenate((n_cov, e_cov)), size)
    local = ndimage.uniform_filter(all_ink.reshape(height, width), size=min(window, width, height), mode="constant")

    return {
        "node_visibility": float(visible.mean()) if n else 1.0,
        "hidden_nodes": float((visible < 0.5).mean()) if n else 0.0,
        "merged_nodes": merged / max(n, 1),
        "ambiguous_edges": len(pairs) / max(m, 1),
        "edge_overlap": edge_overlap,
        "ink_density": float(all_ink.mean()),
        "ink_peak": float(local.max()),
    }


IMAGE_MEASURES = {  # name: True if higher is better, None if descriptive (no better direction)
    "node_visibility": True, "hidden_nodes": False, "merged_nodes": False, "ambiguous_edges": False,
    "edge_overlap": False, "ink_density": None, "ink_peak": False,
}
