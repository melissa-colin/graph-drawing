"""Figure de la slide « Le modèle tour par tour » : les dessins provisoires que le modèle calcule à chaque
tour de la boucle, sur un graphe de test qu'il n'a jamais vu. Même calcul que LayoutModel.forward
(src/model.py), en gardant les positions intermédiaires. Modèle : runs/model_stress_px.pt, celui du tableau
de résultats. Lancer depuis la racine du projet : python docs/jalon2/fig_iterations.py"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.utils import to_undirected

sys.path.insert(0, "src")
from data.input import load_graphml
from model import LayoutModel, knn_edges

GRAPH = Path("data/rome/grafo7007.41.graphml")  # dans le test de CoRe-GD et dans le nôtre
CHECKPOINT = Path("runs/model_stress_px.pt")
STAR = 0  # nœud dont on montre les 8 voisins dans le dessin
OUT = Path("docs/jalon2/iterations.png")

ck = torch.load(CHECKPOINT, map_location="cpu")
config = dict(ck["config"])
config.pop("conv", None)
model = LayoutModel(**config)
model.load_state_dict(ck["state_dict"])
model.eval()

g = load_graphml(GRAPH)
x = g.x if g.x.shape[1] == config["in_features"] else g.x[:, 1:]  # anciens checkpoints : sans la colonne de bruit
edge_index = to_undirected(g.edge_index, num_nodes=g.num_nodes)
batch = torch.zeros(g.num_nodes, dtype=torch.long)

# LayoutModel.forward, en gardant chaque dessin provisoire et ses voisins kNN
drafts, neighbors = [], []
with torch.no_grad():
    h = model.encoder(x)
    for _ in range(model.iterations - 1):
        for conv in model.convs:
            h = conv(h, edge_index)
        positions, _ = model.head(h)
        rewired = knn_edges(positions, batch, model.knn_k)
        drafts.append(positions.clone())
        neighbors.append(rewired)
        h = model.conv_alt(h, rewired)
    for conv in model.convs:
        h = conv(h, edge_index)
    final, _ = model.head(h)

G = nx.Graph()
G.add_nodes_from(range(g.num_nodes))
G.add_edges_from(g.edge_index.t().tolist())
orange, bleu = "#d0661c", "#2563a8"

plt.rcParams.update({"font.family": "DejaVu Sans"})
panels = [(f"Tour {t + 1}", drafts[t], neighbors[t]) for t in range(len(drafts))]
panels.append(("Positions finales", final, None))
fig, axes = plt.subplots(2, 3, figsize=(9.3, 6.6))
axes = axes.flatten()
axes[-1].axis("off")
for ax, (title, P, knn) in zip(axes, panels):
    pos = {i: (float(P[i, 0]), float(P[i, 1])) for i in range(g.num_nodes)}
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#9ca3af", width=0.9)
    if knn is not None:
        src, dst = knn
        mine = [(int(a), STAR) for a, b in zip(src, dst) if int(b) == STAR]  # ses 8 voisins -> ★
        nx.draw_networkx_edges(nx.Graph(mine), pos, ax=ax, edge_color=orange, width=1.6, style="dashed")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=40, node_color=bleu if knn is None else "white",
                           edgecolors="#1f2937", linewidths=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=[STAR], ax=ax, node_size=110, node_color=orange,
                           edgecolors="#1f2937", linewidths=0.8)
    ax.set_xlim(0, 500)
    ax.set_ylim(500, 0)  # y vers le bas, comme une image
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("#d1d5db")
    ax.set_title(title, fontsize=17, fontweight="bold", color="#111827")
fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.06, hspace=0.14)
fig.savefig(OUT, dpi=200, transparent=True)
print("écrit :", OUT)
