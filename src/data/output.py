"""Output format : a drawing, i.e. positions in a fixed-size image.

Unit: pixel and continuous coordinates (not rounded)
Convention: an integer coordinate is the CENTER of a pixel, i.e. pixel i spans
[i - 0.5, i + 0.5] (same as PIL). Origin (0, 0) = center of the top-left pixel.
y axis pointing down.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from data.raster import raster
from data.input import GraphInput


@dataclass
class Layout:
    positions: torch.Tensor # (N, 2) float in [margin, W - margin] x [margin, H - margin] : position of each node in the image, in pixels
    width: int
    height: int


def _edge_pairs(edge_index):
    """Returns a list of (i, j) pairs for each edge in edge_index."""
    if getattr(edge_index, "ndim", 0) != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got shape={tuple(edge_index.shape)}")
    return edge_index.t().tolist()

def _save_png_pil(layout, graph, path, node_color="black", edge_color="black") -> None:
    """Drawing with PIL (existing method)."""
    positions = layout.positions.detach().cpu().tolist()
    img = Image.new("RGB", (layout.width, layout.height), "white")
    draw = ImageDraw.Draw(img)
    
    # 1. edges first (background)
    line_width = max(1, int(round(float(graph.edge_width))))
    for i, j in _edge_pairs(graph.edge_index):
        x0, y0 = positions[i]
        x1, y1 = positions[j]
        draw.line((x0, y0, x1, y1), fill=edge_color, width=line_width)

    # 2. nodes on top (foreground)
    for i, (x, y) in enumerate(positions):
        r = float(graph.node_size[i])
        draw.ellipse((x - r, y - r, x + r, y + r), fill=node_color)

    img.save(path)


def _save_png_raster(layout, graph, path, node_color="black", edge_color="black") -> None:
    """Drawing with our own rasterizer (see raster.py)."""
    positions = layout.positions.detach().cpu().double().tolist()
    # raster.py works with pixel i = [i, i+1] (center at i + 0.5): shift by half a pixel
    # so that an integer coordinate is the center of a pixel, like in the layout convention.
    positions = [(x + 0.5, y + 0.5) for x, y in positions]
    canvas = np.ones((layout.height, layout.width, 3), dtype=np.float64)  # RGB blanc

    edge_width = float(graph.edge_width)
    for i, j in _edge_pairs(graph.edge_index):
        raster.draw_edge(canvas, positions[i], positions[j], edge_width, edge_color)

    for i, c in enumerate(positions):
        raster.draw_node(canvas, c, float(graph.node_size[i]), node_color)

    Image.fromarray(raster.to_uint8(canvas)).save(path)
    

def save_png(layout, graph, path, mode="raster", node_color="black", edge_color="black") -> None:
    """Save a drawing of the graph in a PNG file, using the layout positions.

    mode: "raster" (own rasterizer, default) or "pil" (existing PIL method).
    """
    if mode == "pil":
        _save_png_pil(layout, graph, path, node_color, edge_color)
    elif mode == "raster":
        _save_png_raster(layout, graph, path, node_color, edge_color)
    else:
        raise ValueError(f"unknown mode {mode!r}, expected 'raster' or 'pil'")


def save_positions(layout: Layout, graph: GraphInput, path: Path) -> None:
    """Save the layout positions in a JSON file, using the format specified in the project description."""
    positions = layout.positions.detach().cpu().tolist()
    data = {
        "width": layout.width,
        "height": layout.height,
        "y_axis": "down",
        "node_radius": {node_id: float(size) for node_id, size in zip(graph.node_ids, graph.node_size)},
        "edge_width": float(graph.edge_width),  # a 1-element tensor once separated from a PyG batch
        "positions": {node_id: pos for node_id, pos in zip(graph.node_ids, positions)},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)