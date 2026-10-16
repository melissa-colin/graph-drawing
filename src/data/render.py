"""Rendu différentiable : un dessin devient une image, sans casser le gradient.

Chaque nœud et chaque arête est décrit par une fonction de distance signée (SDF),
transformée en intensité douce : I(p) = sigmoid(-d(p) / eps).
Référence : Iñigo Quilez, https://iquilezles.org/articles/distgradfunctions2d/
"""

import torch

from data.input import GraphInput
from data.output import Layout


def sdf_circle(pixels: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    """Distance signée de chaque pixel au bord de chaque disque.

    pixels  : (H, W, 2) coordonnées des pixels
    centers : (N, 2)
    radii   : (N,)
    retour  : (N, H, W), négatif à l'intérieur du disque
    """
    # TODO(rôle B)
    raise NotImplementedError


def sdf_segment(pixels: torch.Tensor, a: torch.Tensor, b: torch.Tensor, half_width: float) -> torch.Tensor:
    """Distance signée de chaque pixel au bord de chaque arête (segment épais [a, b]).

    pixels : (H, W, 2)
    a, b   : (E, 2) extrémités des arêtes
    retour : (E, H, W)
    """
    # TODO(rôle B)
    raise NotImplementedError


def render(layout: Layout, graph: GraphInput, edge_width: float = 1.0, eps: float = 0.5) -> torch.Tensor:
    """Rend le dessin en image (C, H, W), différentiable par rapport à layout.positions.

    Canaux à décider en équipe, par exemple : 0 = nœuds, 1 = arêtes.
    Attention à la mémoire : de l'ordre de (N + E) x H x W.
    """
    # TODO(rôle B)
    raise NotImplementedError
