
"""Compare PIL and raster rendering."""

from pathlib import Path

import networkx as nx # for the example
import numpy as np
import torch
from PIL import Image

from data.input import GraphInput
from data.output import Layout, save_png


W = H = 400
NODE_R = 7.0
EDGE_W = 1.2
colors = dict(node_color="#2733d6", edge_color="#888888")
OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)


def main():
    # Create graph and layout
    G = nx.karate_club_graph()
    xy = np.array(list(nx.spring_layout(G, seed=0).values()))
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0)) * (W - 50) + 25

    layout = Layout(torch.tensor(xy, dtype=torch.float32), W, H)

    edges = list(G.edges())
    index = {n: i for i, n in enumerate(G.nodes)}

    graph = GraphInput(
        name="karate_club",
        node_ids=[str(n) for n in G.nodes],
        edge_index=torch.tensor(
            [(index[u], index[v]) for u, v in edges],
            dtype=torch.long,
        ).T.contiguous(),
        x=torch.zeros((len(G), 1)),
        node_size=torch.full((len(G),), NODE_R),
        dist=torch.zeros((len(G), len(G))),
        edge_width=EDGE_W,
    )

    # Render both modes using the existing function
    pil_path = OUT / "pil.png"
    raster_path = OUT / "raster.png"

    save_png(layout, graph, pil_path, mode="pil", **colors)
    save_png(layout, graph, raster_path, mode="raster", **colors)

    # Compare side by side
    pil = Image.open(pil_path).convert("RGB")
    raster = Image.open(raster_path).convert("RGB")

    comparison = Image.new("RGB", (2 * W, H), "white")
    comparison.paste(raster, (0, 0))
    comparison.paste(pil, (W, 0))
    comparison.save(OUT / "compare.png")

    print(f"Images saved in {OUT}")


if __name__ == "__main__":
    main()
