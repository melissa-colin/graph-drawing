"""Output format : a drawing, i.e. positions in a fixed-size image.

Unit: pixel and continuous coordinates (not rounded)
origin: top-left corner of the image
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from data.input import GraphInput


@dataclass
class Layout:
    positions: torch.Tensor # (N, 2) float : position of each node in the image, in pixels
    width: int
    height: int


def _edge_pairs(edge_index):
    """Returns a list of (i, j) pairs for each edge in edge_index."""
    if getattr(edge_index, "ndim", 0) != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got shape={tuple(edge_index.shape)}")
    return edge_index.t().tolist()


def save_png(layout: Layout, graph: GraphInput, path: Path) -> None:
    """Save a drawing of the graph in a PNG file, using the layout positions."""
    positions = layout.positions.detach().cpu().tolist()
    img = Image.new("RGB", (layout.width, layout.height), "white")
    draw = ImageDraw.Draw(img)
    for i, (x, y) in enumerate(positions):
        r = float(graph.node_size[i])
        draw.ellipse((x - r, y - r, x + r, y + r), fill="black")

    line_width = max(1, int(round(float(graph.edge_width))))
    for i, j in _edge_pairs(graph.edge_index):
        x0, y0 = positions[i]
        x1, y1 = positions[j]
        draw.line((x0, y0, x1, y1), fill="black", width=line_width)
    img.save(path)


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