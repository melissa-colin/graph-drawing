import torch

from data.input import GraphInput
from data.output import Layout


def edge_crossings(layout: Layout, graph: GraphInput) -> torch.Tensor:
    """Nombre de croisements d'arêtes. Non différentiable : pour l'évaluation.

    À penser : un graphe non planaire ne peut pas descendre à zéro croisement.
    """
    # TODO(rôle C)
    raise NotImplementedError
