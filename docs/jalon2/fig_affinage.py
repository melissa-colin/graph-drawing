"""Figure « L'affinage, de près » : le même graphe de test avant et après l'affinage, rendus par notre
rasteriseur en 500 x 500, avec un zoom x4 sur des défauts corrigés (arête sur un nœud, nœuds qui se
touchent). Choisit, parmi des graphes de test de taille moyenne, celui où le modèle seul a le plus de défauts.
Lancer depuis la racine du projet : python docs/jalon2/fig_affinage.py"""
import sys
import tempfile
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "src")
from data.input import load_dataset, split_dataset
from data.output import Layout, save_png
from infer import load_model
from loss import Loss
from refine import ROUNDS, refine
from train import PHASE_2_WEIGHTS

CHECKPOINT = Path("runs/model_final.pt")
OUT = Path("docs/jalon2/affinage.png")
ZOOM = 64  # côté de la zone zoomée, en pixels de l'image
W = H = 500


def defects(P, g):
    """Arêtes passant sur un nœud qu'elles ne relient pas, et paires de nœuds qui se touchent :
    listes de (position du défaut, type)."""
    r, w = float(g.node_size[0]), float(g.edge_width)
    u, v = g.edge_index
    a, b, p = P[u][:, None], P[v][:, None], P[None]
    ab = b - a
    t = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    dist = (p - (a + t[..., None] * ab)).norm(dim=-1)
    k = torch.arange(P.shape[0])[None]
    on_edge = (dist < r + w / 2) & (k != u[:, None]) & (k != v[:, None])
    found = [[kk] for kk in on_edge.nonzero()[:, 1].tolist()]  # le nœud recouvert par une arête
    D = torch.cdist(P, P)
    found += [[i, j] for i, j in torch.triu(D < 2 * r, diagonal=1).nonzero().tolist()]  # nœuds qui se touchent
    return found  # chaque défaut : la liste des nœuds concernés


def render(P, g):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.png"
        save_png(Layout(P, W, H), g, path, mode="raster", node_color="#2563a8", edge_color="#4b5563")
        return np.array(Image.open(path))


data = Path("data/rome")
graphs = load_dataset(data)
test = [graphs[p.stem] for p in split_dataset(data)[2]]
model = load_model(CHECKPOINT).eval()
loss_fn = Loss(PHASE_2_WEIGHTS)

best = None
for g in [g for g in test if 40 <= g.num_nodes <= 70][:150]:
    with torch.no_grad():
        layout = model(g, ROUNDS)
    found = defects(layout.positions, g)
    if best is None or len(found) > len(best[2]):
        best = (g, layout, found)
g, before, found_before = best
after = refine(before, g, loss_fn)
found_after = defects(after.positions, g)
print(f"{g.name}: {len(found_before)} défauts avant, {len(found_after)} après")

img_before, img_after = render(before.positions, g), render(after.positions, g)
Pb, Pa = before.positions.numpy(), after.positions.numpy()
focus = found_before[0]  # nœuds du défaut montré, suivis d'une image à l'autre
involved = sorted({k for d in found_before for k in d})


def window(P):
    c = P[focus].mean(axis=0)
    return int(np.clip(c[0] - ZOOM / 2, 0, W - ZOOM)), int(np.clip(c[1] - ZOOM / 2, 0, H - ZOOM))


plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
panels = ((axes[0], axes[1], img_before, Pb, "Modèle seul", "avant", len(found_before), "#d62828"),
          (axes[2], axes[3], img_after, Pa, "Modèle + affinage", "après", len(found_after), "#2e7d32"))
for ax_full, ax_zoom, img, P, title, when, n, color in panels:
    x0, y0 = window(P)
    ax_full.imshow(img); ax_full.set_xticks([]); ax_full.set_yticks([])
    ax_full.add_patch(patches.Rectangle((x0, y0), ZOOM, ZOOM, fill=False, ec="#d0661c", lw=2))
    ax_full.set_title(f"{title}\n{n} défaut{'s' if n > 1 else ''} dans ce graphe", fontsize=13, fontweight="bold")
    crop = img[y0:y0 + ZOOM, x0:x0 + ZOOM]
    ax_zoom.imshow(np.kron(crop, np.ones((4, 4, 1), dtype=crop.dtype)), interpolation="nearest")
    ax_zoom.set_xticks([]); ax_zoom.set_yticks([])
    for side in ax_zoom.spines.values():
        side.set_color("#d0661c"); side.set_linewidth(2)
    ax_zoom.set_title(f"Zoom ×4 : {when}", fontsize=13, fontweight="bold")
    for k in involved:  # mêmes nœuds dans les deux zooms : rouge = en défaut, vert = corrigé
        x, y = P[k]
        if x0 <= x < x0 + ZOOM and y0 <= y < y0 + ZOOM:
            ax_zoom.add_patch(patches.Circle(((x - x0) * 4, (y - y0) * 4), 22, fill=False, ec=color, lw=2.5))
fig.text(0.5, 0.03, "Cercles : les nœuds en défaut avant l'affinage (arête qui passe dessus, ou nœuds qui se touchent), "
         "en rouge avant, en vert à leur nouvelle place après.", ha="center", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.1, wspace=0.06)
fig.savefig(OUT, dpi=160)
print("écrit :", OUT)
