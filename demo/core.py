"""Logic of the demo (demo/app.py), without any interface code, so that it can be tested.

A drawing goes: graph -> starting drawing (Kamada-Kawai or our model) -> put in the frame of the chosen image
size, node radius and edge width -> optional refinement in screen space AT THAT SETTING (the team's loss,
weights balanced on this drawing, src/resolution.py) -> optional curved edges (src/curves) -> image measures
(src/metrics/image_quality.py) and a defect map painted in red.
Everything runs on the CPU: one Rome graph takes about a second on a laptop.
"""
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))

from curves.experiment import curve_term_means, curve_weights  # noqa: E402
from curves.optimize import optimize_curves  # noqa: E402
from curves.render import curve_elements, render_curved  # noqa: E402
from data.input import load_graphml  # noqa: E402
from data.output import Layout  # noqa: E402
from data.raster import raster  # noqa: E402
from metrics.batching import static_cache  # noqa: E402
from metrics.image_quality import defect_mask, image_quality  # noqa: E402
from resolution import (anchor, balanced_weights, evaluate, make_loss, rescale_to_frame,  # noqa: E402
                        term_means, with_style)

METHODS = {"kk": "Kamada-Kawai", "ours": "Notre modèle"}
GENERATORS = {
    "grille": lambda n, seed: nx.grid_2d_graph(max(2, int(round(n ** 0.5))), max(2, int(round(n ** 0.5)))),
    "arbre": lambda n, seed: nx.random_labeled_tree(n, seed=seed),
    "cycle": lambda n, seed: nx.cycle_graph(n),
    "géométrique aléatoire": lambda n, seed: nx.random_geometric_graph(n, 1.8 / n ** 0.5, seed=seed),
}
REFINE_FRACTION, REFINE_LR, REFINE_STEPS = 0.1, 1.0, 25
CURVE_FRACTION, CURVE_LR, CURVE_STEPS = 1.0, 1.0, 50


def graph_from_networkx(G, name="graph"):
    """GraphInput of any networkx graph, through a .graphml file and data.input.load_graphml (the features the
    model expects). Only the largest connected component is kept (the stress needs finite distances)."""
    G = nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)))
    structure = nx.Graph()  # nodes and edges only: attributes (e.g. positions as lists) cannot go to GraphML
    structure.add_nodes_from(range(G.number_of_nodes()))
    structure.add_edges_from(G.edges())
    G = structure
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.graphml"
        nx.write_graphml(G, path)
        return load_graphml(path)


def generate(kind, n, seed=0):
    return graph_from_networkx(GENERATORS[kind](int(n), int(seed)), f"{kind}-{n}")


def degree_radii(graph, radius):
    """(N,) node radii growing with the degree: radius * sqrt(degree / mean degree), within [radius / 2, 2 radius]
    (hubs drawn bigger, a common style that only a screen-space method can take into account)."""
    degree = torch.bincount(graph.edge_index.reshape(-1), minlength=graph.num_nodes).float()
    return (radius * (degree / degree.mean().clamp_min(1e-9)).sqrt()).clamp(radius / 2, 2 * radius)


def kamada_kawai(graph):
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    G.add_edges_from(graph.edge_index.t().tolist())
    p = nx.kamada_kawai_layout(G)
    return torch.tensor(np.array([p[i] for i in range(graph.num_nodes)]), dtype=torch.float32)


def load_model(checkpoint):
    from infer import load_model as load
    return load(Path(checkpoint)) if checkpoint else None


def start_drawing(graph, method, model=None):
    if method == "kk":
        return kamada_kawai(graph)
    if model is None:
        raise ValueError("our model needs a checkpoint (python demo/app.py --checkpoint runs/model.pt)")
    with torch.no_grad():
        return model(graph, 12).positions  # refine.ROUNDS rounds, as at inference


def refine_with_snapshots(positions, graph, size, loss_fn, steps, lr, margin):
    """src/refine.refine (Adam on the positions, kept in the frame), keeping the positions after every step
    for the animation. Same result as refine.refine (tested)."""
    P = positions.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([P], lr=lr)
    frames = [P.detach().clone()]
    with torch.enable_grad(), static_cache(graph):
        for _ in range(steps):
            optimizer.zero_grad()
            loss = loss_fn(Layout(P, size, size), graph)[0]
            if not loss.requires_grad:
                break
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                P.clamp_(margin, size - margin)
            frames.append(P.detach().clone())
    return P.detach(), frames


@dataclass
class Drawing:
    graph: object  # styled GraphInput
    size: int
    positions: torch.Tensor
    offsets: torch.Tensor
    frames: list = field(default_factory=list)
    seconds: dict = field(default_factory=dict)
    measures: dict = field(default_factory=dict)


def draw(graph, method, size, radius, width, refine=True, curves=False, model=None, steps=REFINE_STEPS,
         varied=False):
    """The whole chain for one graph. Returns a Drawing with its measures.
    varied: node radii by degree (degree_radii) instead of `radius` for every node."""
    seconds = {}
    t = time.perf_counter()
    start = start_drawing(graph, method, model)
    seconds["départ"] = time.perf_counter() - t
    radii = degree_radii(graph, radius) if varied else torch.full((graph.num_nodes,), float(radius))
    margin = float(radii.max())
    P = rescale_to_frame(start, size, margin)
    g = with_style(graph, radius, width)
    g.node_size = radii
    g = anchor(g, P)  # stress at the scale of this drawing (see src/resolution.py)
    frames = [P.clone()]
    if refine and steps > 0:
        t = time.perf_counter()
        weights = balanced_weights(term_means([(P, g)], "screen", size), REFINE_FRACTION)
        P, frames = refine_with_snapshots(P, g, size, make_loss("screen", weights), steps, REFINE_LR, margin)
        seconds["affinage"] = time.perf_counter() - t
    offsets = torch.zeros(g.edge_index.shape[1], 2)
    if curves:
        t = time.perf_counter()
        weights = curve_weights(curve_term_means([(P, g)], size), CURVE_FRACTION)
        _, offsets = optimize_curves(Layout(P, size, size), g, weights, steps=CURVE_STEPS, lr=CURVE_LR)
        seconds["courbes"] = time.perf_counter() - t
    d = Drawing(g, size, P, offsets, frames, seconds)
    d.measures = measure(d)
    return d


def _edges(d):
    return curve_elements(d.positions, d.offsets, d.graph, d.size, d.size) if d.offsets.abs().sum() > 0 else None


def measure(d):
    edges = _edges(d)
    m = evaluate(d.positions, d.graph, d.size) if edges is None else image_quality(d.positions, d.graph, d.size, d.size, edges=edges)
    if edges is not None:
        from curves.losses import count_curve_crossings
        from resolution import scale_normalized_stress
        m["stress"] = scale_normalized_stress(d.positions, d.graph)
        m["crossings"] = count_curve_crossings(Layout(d.positions, d.size, d.size), d.offsets, d.graph)
    return m


MEASURE_LABELS = {"hidden_nodes": "Nœuds cachés (part)", "merged_nodes": "Nœuds fusionnés (part)",
                  "ambiguous_edges": "Arêtes ambiguës (par arête)", "edge_overlap": "Arêtes confondues (part)",
                  "node_visibility": "Visibilité moyenne des nœuds", "stress": "Stress", "crossings": "Croisements"}


def measures_table(drawings):
    """Rows [measure, value for each drawing] for a table (labels in French, as in the slides)."""
    return [[label] + [round(float(d.measures[key]), 3) for d in drawings] for key, label in MEASURE_LABELS.items()]


def render(d, scale=2, defects=True, positions=None):
    """PIL image of the drawing, enlarged `scale` times (nearest pixel: every screen pixel stays visible),
    defects in red. positions: another frame of the animation (defects then recomputed on it)."""
    P = d.positions if positions is None else positions
    highlight = None
    if defects:
        from scipy import ndimage
        edges = curve_elements(P, d.offsets, d.graph, d.size, d.size) if d.offsets.abs().sum() > 0 else None
        mask = defect_mask(P, d.graph, d.size, d.size, edges=edges)
        highlight = ndimage.binary_dilation(mask > 0, iterations=1).astype(float)
    img = render_curved(P, d.offsets, d.graph, d.size, d.size, highlight=highlight)
    image = Image.fromarray(raster.to_uint8(img))
    return image.resize((d.size * scale, d.size * scale), Image.NEAREST) if scale != 1 else image


def animation(d, path, scale=2, every=1, duration_ms=120):
    """GIF of the refinement, frame by frame, defects in red (straight edges: the curves come after)."""
    straight = Drawing(d.graph, d.size, d.positions, torch.zeros_like(d.offsets))
    frames = [render(straight, scale, True, P) for P in d.frames[::every]]
    if d.offsets.abs().sum() > 0:
        frames.append(render(d, scale, True))  # last frame: the curves
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return path
