import torch

from data.input import GraphInput
from data.output import Layout


def node_overlap_geometric(layout: Layout, graph: GraphInput) -> torch.Tensor:
    """Chevauchement des nœuds calculé en géométrie : pénalise ||c_i - c_j|| < r_i + r_j. Différentiable."""
    # TODO(rôle C)
    raise NotImplementedError


def node_overlap_screen(image: torch.Tensor) -> torch.Tensor:
    """Chevauchement des nœuds calculé sur l'image rendue (render.py). Différentiable.

    Piste : à partir de la carte somme S = sum_i I_i plutôt que paire par paire.
    """
    # TODO(rôle C)
    raise NotImplementedError
