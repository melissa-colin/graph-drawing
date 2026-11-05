import torch

def node_sdf_torch(px, py, c, r):
    """Signed distance to a disk. Negative inside."""
    dx = px - c[0]
    dy = py - c[1]
    # distance - radius + espilon (for retropagation)
    return torch.sqrt(dx.square() + dy.square() + 1e-12) - r


def edge_sdf_torch(px, py, a, b, w):
    """Signed distance to a rectangle edge with flat caps. Negative inside."""
    # Direction and length of the edge
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length_ab = torch.sqrt(dx.square() + dy.square() + 1e-12)

    # unit vector along the edge
    nx = dx / length_ab
    ny = dy / length_ab

    # midpoint of the rectangle
    mx = (a[0] + b[0]) / 2
    my = (a[1] + b[1]) / 2

    # signed offsets from the rectangle boundaries
    # |projection of vector MP onto the edge direction| - minus half the edge length
    along_edge = ((px - mx) * nx + (py - my) * ny).abs() - length_ab / 2 # direction of the length
    across_edge = (-(px - mx) * ny + (py - my) * nx).abs() - w / 2  # direction of the width

    # distance outside the rectangle (to the nearest edge or corner).
    outside = torch.sqrt(torch.clamp(along_edge, min=0).square() + torch.clamp(across_edge, min=0).square() + 1e-12)

    # signed distance inside the rectangle: negative distance to the nearest edge.
    inside = torch.minimum(torch.maximum(along_edge, across_edge), along_edge.new_zeros(()))

    return outside + inside
