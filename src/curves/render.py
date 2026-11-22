"""Rendering of curved edges: pixel coverage by supersampling, PNG output, and the elements that
metrics/image_quality.image_quality takes for curved drawings (its `edges` argument).

Coverage of a pixel = share of an n x n grid of points inside the pixel that lie in the stroke (distance to
the polyline <= w / 2), n = 4 by default: error at most 1 / 16 per boundary pixel, unbiased on average.
Conventions of data/output.py: an integer coordinate is the center of a pixel.
"""
import numpy as np
import torch
from PIL import Image

from curves.bezier import K, sample
from data.raster import raster
from metrics.image_quality import Element, band_pixels


def _polyline_distance(x, y, points):
    """Distance from the points (x, y) (any shape) to the polyline points (k + 1, 2), numpy."""
    a, b = points[:-1], points[1:]
    ab = b - a
    px, py = x[..., None], y[..., None]
    t = np.clip(((px - a[:, 0]) * ab[:, 0] + (py - a[:, 1]) * ab[:, 1]) / np.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
    return np.hypot(px - (a[:, 0] + t * ab[:, 0]), py - (a[:, 1] + t * ab[:, 1])).min(axis=-1)


def curve_element(points, edge_width, width, height, supersample=4):
    """Element (sparse coverage) of the stroke of width edge_width along points (k + 1, 2), Layout coordinates."""
    pts = np.asarray(points, dtype=np.float64) + 0.5  # raster convention: pixel i spans [i, i + 1]
    X, Y = band_pixels(pts, edge_width / 2 + 2.0, width, height)
    sub = (np.arange(supersample) + 0.5) / supersample
    sx = X[:, None, None] + sub[None, None, :]
    sy = Y[:, None, None] + sub[None, :, None]
    inside = _polyline_distance(np.broadcast_to(sx, (len(X), supersample, supersample)),
                                np.broadcast_to(sy, (len(X), supersample, supersample)), pts) <= edge_width / 2
    cov = inside.mean(axis=(1, 2))
    keep = cov > 0
    return Element((Y[keep] * width + X[keep]).astype(np.int64), cov[keep])


def curve_elements(positions, offsets, graph, width, height, k=K, supersample=4):
    """One Element per edge of graph.edge_index, for image_quality(..., edges=...)."""
    with torch.no_grad():
        pts = sample(torch.as_tensor(positions, dtype=torch.float64), graph.edge_index,
                     torch.as_tensor(offsets, dtype=torch.float64), k).numpy()
    return [curve_element(p, float(graph.edge_width), width, height, supersample) for p in pts]


def save_png_curved(positions, offsets, graph, path, width, height, node_color="#2563a8", edge_color="#4b5563",
                    highlight=None):
    """Drawing with curved edges (edges first, nodes on top), like data/output.save_png.
    highlight: optional (H, W) array in [0, 1], painted in red over the drawing (defects, for the demo)."""
    canvas = render_curved(positions, offsets, graph, width, height, node_color, edge_color, highlight)
    Image.fromarray(raster.to_uint8(canvas)).save(path)


def render_curved(positions, offsets, graph, width, height, node_color="#2563a8", edge_color="#4b5563",
                  highlight=None):
    """(H, W, 3) float image in [0, 1] of the drawing with curved edges."""
    canvas = np.ones((height, width, 3))
    flat = canvas.reshape(-1, 3)
    color = raster.to_rgb(edge_color)
    for e in curve_elements(positions, offsets, graph, width, height):
        c = e.coverage[:, None]
        flat[e.pixels] = flat[e.pixels] * (1 - c) + color * c
    P = torch.as_tensor(positions).detach().cpu().double().tolist()
    for i, (x, y) in enumerate(P):
        raster.draw_node(canvas, (x + 0.5, y + 0.5), float(graph.node_size[i]), node_color)
    if highlight is not None:
        h = np.clip(highlight, 0, 1)[..., None]
        canvas[:] = canvas * (1 - 0.7 * h) + raster.to_rgb("#e34948") * 0.7 * h
    return canvas
