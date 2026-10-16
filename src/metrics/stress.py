import torch

from data.input import GraphInput
from data.output import Layout


def stress(layout: Layout, graph: GraphInput) -> torch.Tensor:
    """Écart entre distances dessinées et distances dans le graphe (graph.dist). Différentiable.

    sum_{i<j} (||x_i - x_j|| - d_ij)^2 / d_ij^2, après remise du dessin à la meilleure échelle.
    """
    # TODO(rôle C)
    raise NotImplementedError
