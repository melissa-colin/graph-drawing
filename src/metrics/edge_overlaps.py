import math
import torch
from data.raster.sdf import edge_coverage_torch, coverage_margin
from .metrics_helpers import overlap_in_box

def edge_overlaps(layout, graph, softness=0.5):
    pos = layout.positions            # (N, 2)
    u, v = graph.edge_index           # (E,), (E,)
    edge_width = float(graph.edge_width)
    H, W = layout.height, layout.width
    margin = coverage_margin(softness)

    # bounding box of each edge
    edge_start_positions, edge_end_positions = pos.detach()[u], pos.detach()[v]
    edge_box_min = torch.minimum(edge_start_positions, edge_end_positions) - edge_width / 2
    edge_box_max = torch.maximum(edge_start_positions, edge_end_positions) + edge_width / 2

    # check only nearby edges: DEPENDS ON THE SHAPE
    boxes_touch = (
        (edge_box_min[:, None] - margin < edge_box_max[None, :])
        & (edge_box_min[None, :] - margin < edge_box_max[:, None])
    ).all(dim=-1)
    # edges sharing a node, compare : start/start, start/end, end/start, end/end
    share_node = (u[:, None] == u[None, :]) | (u[:, None] == v[None, :]) \
               | (v[:, None] == u[None, :]) | (v[:, None] == v[None, :])

    # close edges without common node
    close = torch.triu(boxes_touch & (share_node == False), diagonal=1)

    loss = pos.new_zeros(())
    for i, j in close.nonzero().tolist():
        # define the area within which the overlap will be sought
        overlap_box_min = torch.maximum(edge_box_min[i], edge_box_min[j]) - margin
        overlap_box_max = torch.minimum(edge_box_max[i], edge_box_max[j]) + margin

        # check the coverage
        loss = loss + overlap_in_box(
           overlap_box_min, overlap_box_max, W, H, pos,
            lambda X, Y: edge_coverage_torch(X, Y, pos[u[i]], pos[v[i]], edge_width, softness),
            lambda X, Y: edge_coverage_torch(X, Y, pos[u[j]], pos[v[j]], edge_width, softness),
        )

    return loss
