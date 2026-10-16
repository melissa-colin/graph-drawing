"""Modèle : un graphe en entrée, des positions écran en sortie."""

import torch.nn as nn

from data.input import GraphInput
from data.output import Layout


class LayoutModel(nn.Module):
    """GNN qui prédit la position de chaque nœud dans une image width x height."""

    def __init__(self, in_features: int, hidden: int, width: int, height: int):
        super().__init__()
        self.width = width
        self.height = height
        # TODO(rôle A) : couches du modèle

    def forward(self, graph: GraphInput) -> Layout:
        """Les positions renvoyées doivent être continues et rester dans [0, width] x [0, height]."""
        # TODO(rôle A)
        raise NotImplementedError
