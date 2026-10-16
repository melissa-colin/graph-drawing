"""Format de sortie commun : un dessin, c'est-à-dire des positions dans une image de taille fixée."""

from dataclasses import dataclass
from pathlib import Path

import torch

from data.input import GraphInput


@dataclass
class Layout:
    positions: torch.Tensor  # (N, 2) float : coordonnées continues en pixels, dans [0, width] x [0, height]
    width: int               # largeur de l'image, en pixels
    height: int              # hauteur de l'image, en pixels


def save_png(layout: Layout, graph: GraphInput, path: Path) -> None:
    """Enregistre le dessin en image, pour le regarder."""
    # TODO(rôle A)
    raise NotImplementedError


def save_positions(layout: Layout, graph: GraphInput, path: Path) -> None:
    """Enregistre les positions dans un fichier (format à décider : JSON, CSV...)."""
    # TODO(rôle A)
    raise NotImplementedError
