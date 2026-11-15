"""Figure de la slide « Le prétraitement, sur un exemple » : ce que load_graphml calcule pour un petit graphe
de Rome. Les valeurs viennent du vrai code (src/data/input.py) ; seul le placement des nœuds (Kamada-Kawai)
sert à l'affichage. Lancer depuis la racine du projet : python docs/jalon2/fig_pretraitement.py"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, "src")
from data.input import load_graphml

GRAPH = Path("data/rome/grafo1000.14.graphml")
SOURCE = 0  # nœud dont on montre les distances et la ligne de x
OUT = Path("docs/jalon2/pretraitement.png")

g = load_graphml(GRAPH)
G = nx.Graph()
G.add_nodes_from(range(g.num_nodes))
G.add_edges_from(g.edge_index.t().tolist())
pos = nx.kamada_kawai_layout(G)
x = g.x.numpy()
noise, degree, pe = x[:, 0], x[:, 1], x[:, 3:]
i, j = g.dist_index  # paires i < j et leur distance (dist_index, dist_value)
dist = np.zeros(g.num_nodes, dtype=int)
for a, b, d in zip(i.tolist(), j.tolist(), g.dist_value.tolist()):
    if SOURCE in (a, b):
        dist[b if a == SOURCE else a] = int(d)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(1, 5, figsize=(13, 3.9))
bleu, orange, gris = "#2563a8", "#d0661c", "#6b7280"


def panel(ax, title, colors=None, labels=None, cmap=None, vmin=None, vmax=None, highlight=None):
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#9ca3af", width=1.2)
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=420, node_color=colors if colors is not None else "white",
                                   cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="#1f2937", linewidths=1.0)
    if highlight is not None:
        nx.draw_networkx_nodes(G, pos, nodelist=[highlight], ax=ax, node_size=420, node_color="none",
                               edgecolors=orange, linewidths=3)
    if labels is not None:
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=11, font_weight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", color="#111827")
    ax.axis("off")
    return nodes


panel(axes[0], "Graphe d'entrée")
panel(axes[1], "Degré", colors="#dbeafe", labels={i: int(degree[i]) for i in range(g.num_nodes)})
panel(axes[2], "Bruit aléatoire", colors=noise, cmap="Greys", vmin=0, vmax=1.3)
panel(axes[3], "Vecteur propre n°1\ndu Laplacien (sur 8)", colors=pe[:, 0], cmap="coolwarm",
      vmin=-abs(pe[:, 0]).max(), vmax=abs(pe[:, 0]).max())
panel(axes[4], "Distances depuis ★\n(pour la loss)", colors="#fde8d7",
      labels={i: ("★" if i == SOURCE else int(dist[i])) for i in range(g.num_nodes)}, highlight=SOURCE)
for ax in axes:
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")

row = x[SOURCE]
text = ("x du nœud ★ = [ bruit {:.2f} | degré {:.0f} | 1 | Laplacien {} ]"
        .format(row[0], row[1], "  ".join(f"{v:+.2f}" for v in row[3:])))
fig.text(0.5, 0.05, text, ha="center", fontsize=14, family="DejaVu Sans Mono", color="#111827",
         bbox=dict(boxstyle="round,pad=0.5", fc="#eef4fb", ec=bleu))
fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.2, wspace=0.05)
fig.savefig(OUT, dpi=200, transparent=True)
print("écrit :", OUT)
