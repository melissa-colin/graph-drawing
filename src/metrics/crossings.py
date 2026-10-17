import torch

from data.input import GraphInput
from data.output import Layout


def edge_crossings(layout: Layout, graph: GraphInput) -> torch.Tensor:
    """Number of edge crossings. Not differentiable: for evaluation only.

    Keep in mind: a non-planar graph cannot get down to zero crossings.
    """
    # TODO(role C)
    raise NotImplementedError
