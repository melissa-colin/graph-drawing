"""Format de sortie commun : un dessin, c'est-à-dire des positions dans une image de taille fixée.

Conventions (valables partout : modèle, rendu, métriques, export) :
- unité : le pixel ; les coordonnées sont continues (pas arrondies) ;
- origine en haut à gauche de l'image, x vers la droite, y vers le BAS (convention image) ;
- positions[i] est la position du nœud d'indice i, c'est-à-dire du nœud graph.node_ids[i].
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from data.input import GraphInput


@dataclass
class Layout:
    positions: torch.Tensor
    width: int
    height: int


def _edge_pairs(edge_index):
    """Retourne les arêtes sous forme [(i, j), ...] pour edge_index en (2, E)."""
    if getattr(edge_index, "ndim", 0) != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index doit avoir la forme (2, E), got shape={tuple(edge_index.shape)}")
    return edge_index.t().tolist()


def save_png(layout: Layout, graph: GraphInput, path: Path) -> None:
    """Enregistre le dessin en image, pour le regarder."""
    positions = layout.positions.detach().cpu().tolist()
    img = Image.new("RGB", (layout.width, layout.height), "white")
    draw = ImageDraw.Draw(img)
    for i, (x, y) in enumerate(positions):
        r = float(graph.node_size[i])
        draw.ellipse((x - r, y - r, x + r, y + r), fill="black")

    line_width = max(1, int(round(graph.edge_width)))
    for i, j in _edge_pairs(graph.edge_index):
        x0, y0 = positions[i]
        x1, y1 = positions[j]
        draw.line((x0, y0, x1, y1), fill="black", width=line_width)
    img.save(path)


def save_positions(layout: Layout, graph: GraphInput, path: Path) -> None:
    """Enregistre le dessin en JSON, avec tout ce qu'il faut pour le réafficher à l'identique."""
    positions = layout.positions.detach().cpu().tolist()
    data = {
        "width": layout.width,
        "height": layout.height,
        "y_axis": "down",
        "node_radius": {node_id: float(size) for node_id, size in zip(graph.node_ids, graph.node_size)},
        "edge_width": graph.edge_width,
        "positions": {node_id: pos for node_id, pos in zip(graph.node_ids, positions)},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    exemple_graph = GraphInput(
        name="exemple",
        node_ids=["n0", "n1", "n2", "n3"],
        edge_index=torch.tensor([[0, 1, 0], [1, 2, 3]]),
        x=torch.rand(4, 5),
        node_size=torch.tensor([5.0, 6.0, 7.0, 8.0]),
        dist=torch.rand(4, 4),
        edge_width=2.0,
    )
    exemple_layout = Layout(
        positions=torch.tensor([[100.0, 100.0], [200.0, 150.0], [150.0, 250.0], [300.0, 300.0]]),
        width=400,
        height=400,
    )
    save_png(exemple_layout, exemple_graph, Path("exemple.png"))
    save_positions(exemple_layout, exemple_graph, Path("exemple.json"))