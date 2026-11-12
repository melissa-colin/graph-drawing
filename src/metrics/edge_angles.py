
import math
import torch

from data.input import GraphInput
from data.output import Layout

EPS = 1e-8

def edge_angles(layout: Layout, graph: GraphInput) -> torch.Tensor:
    pos = layout.positions
    # edge in every direction
    start_nodes, end_nodes = graph.edge_index_bidirectional.to(pos.device)

    # edge from the same node
    same_node = (start_nodes[:, None] == start_nodes[None, :]).triu(diagonal=1)
    a, b = same_node.nonzero(as_tuple=True)

    # all nodes of degree 1
    if a.numel() == 0:
        return torch.zeros((), device=pos.device, dtype=pos.dtype)

    # compute angles
    # a . b = ||a|| * ||b|| * cos(ab) & C = ||a|| ||b|| sinf(ab) => cross/dot = tan
    vec = pos[end_nodes] - pos[start_nodes]
    cross = vec[a, 0] * vec[b, 1] - vec[a, 1] * vec[b, 0]
    dot = (vec[a] * vec[b]).sum(dim=1)
    theta = torch.atan2(cross.abs()  + EPS, dot)

    # compute degrees
    degrees = torch.bincount(start_nodes)
    d_max = degrees.max().to(dtype=pos.dtype)
    theta_g = 2 * math.pi / d_max

    if theta.numel() == 0:
        return pos.new_tensor(0.0)
    
    return 1.0 - theta.min() / theta_g
