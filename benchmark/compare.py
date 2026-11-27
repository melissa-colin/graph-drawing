"""Benchmark: random / our model (alone, then + refinement) / Kamada-Kawai / DeepGD / CoRe-GD on the 1000 test graphs of Rome.

Our split (src/data/input.split_dataset) is the one of CoRe-GD and DeepGD: no learned model saw these graphs.
DeepGD and CoRe-GD use the pretrained Rome checkpoints shipped with the CoRe-GD repo.
For each method: quality (scale-normalized stress, Ahmed et al. 2026, divided by the number of pairs; edge
crossings; share of overlapping node pairs), size (largest side of the drawing; median distance from a node
to its nearest node, in the 500 x 500 frame) and cost (parameters, time and peak GPU memory per graph).
Also renders the drawings of the five methods on the same three graphs (30, 60 and 100 nodes) with our
rasterizer, in the 500 x 500 frame, and assembles them in one sheet.

Setup once:  bash benchmark/setup_baselines.sh
Run from the project root:  python benchmark/compare.py runs/model.pt
Also the quality measures of (GD)^2 (src/metrics/evaluation.py), to compare with the literature.
Outputs in benchmark/out/: results.md (tables), positions.pt (all drawings), renders/ (one 500 x 500 PNG per method and graph), sheet.png
"""
import json, random, sys, time, types
from pathlib import Path
import numpy as np, networkx as nx, torch
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.utils import scatter as pyg_scatter, to_undirected

HERE = Path(__file__).resolve().parent
EXT = HERE / "ext"
OUT = HERE / "out"
# --- pure-PyTorch stand-ins for two compiled extensions that are not installed here
ts = types.ModuleType("torch_scatter")
ts.scatter = lambda src, index, dim=0, out=None, dim_size=None, reduce="sum": pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)
sys.modules["torch_scatter"] = ts
tc = types.ModuleType("torch_cluster")
def knn(x, y, k, batch_x=None, batch_y=None, **kw):   # for each y: its k nearest x (self included), like torch_cluster
    d = torch.cdist(y, x)
    if batch_x is not None:
        d = d.masked_fill(batch_y[:, None] != batch_x[None, :], float("inf"))
    k = min(k, x.shape[0]); _, idx = d.topk(k, dim=1, largest=False)
    row = torch.arange(y.shape[0], device=y.device)[:, None].expand(-1, k)
    return torch.stack((row.reshape(-1), idx.reshape(-1)))
tc.knn = knn
sys.modules["torch_cluster"] = tc
sys.path[:0] = [str(EXT / "CoRe-GD"), str(EXT / "deepgd"), str(HERE.parent / "src"), str(HERE)]

from data.input import graph_input, load_dataset, split_dataset
from infer import load_model
from neuraldrawer.network.model import get_model
from neuraldrawer.network.preprocessing import preprocess_dataset
import neuraldrawer.network.preprocessing as core_prep
import deepgd

# Recent PyG transforms must define forward(); theirs overrides __call__ directly, which still does the work.
core_prep.AddLaplacian.forward = lambda self, data: self(data)
core_prep.AddLaplacian.__abstractmethods__ = frozenset()

W = H = 500; M = 4.0; R = 4.0
dev = "cuda" if torch.cuda.is_available() else "cpu"

# --- metrics
def stress(P, g):   # normalized stress after the best uniform scaling
    i, j = g.dist_index; d = g.dist_value; e = (P[i] - P[j]).norm(dim=1)
    w = d ** -2; s = (w * e * d).sum() / (w * e * e).sum()
    return float((w * (s * e - d) ** 2).sum() / len(d))
def overlaps(P):    # share of node pairs whose disks overlap
    D = torch.cdist(P, P); n = len(P); return float(((D < 2 * R).sum() - n) / (n * (n - 1)))
def crossings(P, g):   # pairs of edges that cross (edges sharing a node skipped)
    e = g.edge_index.t(); a, b = P[e[:, 0]], P[e[:, 1]]
    def orient(p, q, r): return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    A1, B1 = a[:, None], b[:, None]; A2, B2 = a[None], b[None]
    cross = (orient(A1, B1, A2) * orient(A1, B1, B2) < 0) & (orient(A2, B2, A1) * orient(A2, B2, B1) < 0)
    sh = (e[:, None, 0] == e[None, :, 0]) | (e[:, None, 0] == e[None, :, 1]) | (e[:, None, 1] == e[None, :, 0]) | (e[:, None, 1] == e[None, :, 1])
    return int((cross & ~sh).triu(1).sum())
def to_frame(X):   # uniform rescale into the frame, as for Kamada-Kawai
    X = X.float().cpu(); X = X - X.mean(0)
    return X * (min(W, H) / 2 - M) / X.abs().max().clamp_min(1e-9) + torch.tensor([W / 2, H / 2])
def nxg(g):
    G = nx.Graph(); G.add_nodes_from(range(g.num_nodes)); G.add_edges_from(g.edge_index.t().tolist()); return G

# --- our model
if len(sys.argv) < 2:
    sys.exit("usage: python benchmark/compare.py CHECKPOINT")
CHECKPOINT = sys.argv[1]
ours = load_model(Path(CHECKPOINT)).to(dev)
cfg = ours.config
print("our checkpoint:", CHECKPOINT, "| head:", cfg["head"],
      "| loss weights:", torch.load(CHECKPOINT, map_location="cpu").get("loss_weights", "not recorded"))
from loss import Loss
from metrics.evaluation import MEASURES
from refine import ROUNDS, STEPS, draw
from train import REFINE_WEIGHTS
refine_loss = Loss(weights=REFINE_WEIGHTS)
def ours_pos(g, steps=0):
    """Our model with ROUNDS rounds, then `steps` refinement steps on the positions (src/refine.py).
    Preprocessing included, as for the other methods: features from the bare graph (degree, Laplacian
    eigenvectors, noise), plus the graph distances when refining (the stress of the refinement needs them)."""
    h = graph_input(nxg(g), g.name, pe_dim=g.pe_dim, distances=steps > 0)
    if h.x.shape[1] == cfg["in_features"] + 1:  # checkpoints from before the noise column
        h.x = h.x[:, 1:]
    return draw(ours, h.to(dev), refine_loss, ROUNDS, steps).positions.cpu()
def random_pos(g, rng):
    return torch.tensor([[rng.uniform(M, W - M), rng.uniform(M, H - M)] for _ in range(g.num_nodes)])
def kk_pos(g):
    p = nx.kamada_kawai_layout(nxg(g)); return to_frame(torch.tensor(np.array([p[i] for i in range(g.num_nodes)])))

# --- CoRe-GD
core_cfg = types.SimpleNamespace(**json.load(open(EXT / "CoRe-GD/configs/config_rome.json")))
core_cfg.use_cupy = False
core = get_model(core_cfg)
core.load_state_dict(torch.load(EXT / "CoRe-GD/checkpoints/core_rome.pt", map_location="cpu")); core.eval().to(dev)
def core_pos(g):
    d = Data(x=torch.ones(g.num_nodes, 1), edge_index=to_undirected(g.edge_index, num_nodes=g.num_nodes))
    d = preprocess_dataset([d], core_cfg)[0]
    b = pyg.data.Batch.from_data_list([d]).to(dev)
    with torch.no_grad(): return to_frame(core(b, 20))   # 20 iterations, as in their evaluation script

# --- DeepGD (initialized with PivotMDS; the authors computed it with R graphlayouts)
def pivot_mds(G, k=10):   # Brandes & Pich 2006
    n = G.number_of_nodes(); D = np.array(nx.floyd_warshall_numpy(G, nodelist=range(n)))
    k = min(k, n); piv = [0]
    while len(piv) < k: piv.append(int(np.argmax(D[:, piv].min(1))))
    C = D[:, piv] ** 2
    C = -0.5 * (C - C.mean(0, keepdims=True) - C.mean(1, keepdims=True) + C.mean())
    U, s, _ = np.linalg.svd(C, full_matrices=False); return U[:, :2] * s[:2]
dgd = deepgd.DeepGD()
dgd.load_state_dict(torch.load(EXT / "CoRe-GD/deepgd_checkpoints/rome.pt", map_location="cpu")); dgd.eval().to(dev)
def deepgd_pos(g):
    G = nxg(g); n = g.num_nodes
    apsp = dict(nx.all_pairs_shortest_path_length(G))
    full, dd = zip(*[((u, v), d) for u in apsp for v, d in apsp[u].items()])
    fe, d = pyg.utils.remove_self_loops(*to_undirected(torch.tensor(full).T, torch.tensor(dd)))
    d = d.float(); init = torch.tensor(pivot_mds(G), dtype=torch.float)
    data = Data(x=init, init_pos=init, edge_index=fe, edge_attr=torch.stack([d, 1 / d ** 2], -1), full_edge_index=fe, d=d,
                edge_weight=torch.ones(fe.shape[1]),   # weight 1 on every pair: no effect, their layer multiplies by it
                raw_edge_index=to_undirected(g.edge_index, num_nodes=n), n=n)
    b = pyg.data.Batch.from_data_list([data]).to(dev)
    with torch.no_grad(): return to_frame(dgd(b))

# --- evaluation on the graphs unseen by every model
def timed(f, g):
    """Positions, wall time (s) and peak GPU memory (MiB) of one call."""
    if dev == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    P = f(g)
    if dev == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t
    memory = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else float("nan")
    return P, seconds, memory

DATA = HERE.parent / "data" / "rome"
gs = load_dataset(DATA)
graphs = [gs[path.stem] for path in split_dataset(DATA)[2]]  # test split, shared with CoRe-GD and DeepGD
rng = random.Random(0)
methods = [("Aléatoire", lambda g: random_pos(g, rng)), ("Notre modèle", ours_pos),
           ("Notre modèle + affinage", lambda g: ours_pos(g, STEPS)), ("Kamada-Kawai", kk_pos),
           ("DeepGD", deepgd_pos), ("CoRe-GD", core_pos)]
params = {"Aléatoire": 0, "Kamada-Kawai": 0, "Notre modèle": sum(p.numel() for p in ours.parameters()),
          "Notre modèle + affinage": sum(p.numel() for p in ours.parameters()),
          "DeepGD": sum(p.numel() for p in dgd.parameters()), "CoRe-GD": sum(p.numel() for p in core.parameters())}
for k, f in methods[1:]:  # warm-up: first CUDA calls are slow
    f(graphs[0])
res = {k: [] for k, _ in methods}
literature = {k: [] for k, _ in methods}  # (GD)^2 quality measures, src/metrics/evaluation.py
cost = {k: [] for k, _ in methods}
positions = {k: {} for k, _ in methods}
for g in graphs:
    for k, f in methods:
        P, seconds, memory = timed(f, g)
        positions[k][g.name] = P
        D = torch.cdist(P.float(), P.float()); D.fill_diagonal_(float("inf"))
        # size checks: the stress is scale-free, so a drawing can shrink without any quality metric noticing
        span = float((P.max(0).values - P.min(0).values).max())
        nearest = float(D.min(1).values.median())
        res[k].append((stress(P, g), overlaps(P), crossings(P, g), span, nearest))
        literature[k].append([f(P.double().cpu(), g.cpu()) for f, _ in MEASURES.values()])
        cost[k].append((seconds, memory))
OUT.mkdir(exist_ok=True)
baseline_memory = min(np.nanmin([c[1] for c in cost[k]]) for k in cost)  # weights already loaded on the GPU
lines = [f"Test graphs: {len(graphs)} (Rome, split shared with CoRe-GD and DeepGD) | device: {dev} | ours: {CHECKPOINT}", "",
         "| method | stress | crossings | overlapping nodes | drawing size | nearest node | parameters | time / graph | peak GPU memory |",
         "|---|---|---|---|---|---|---|---|---|"]
for k, v in res.items():
    memory = np.nanmax([c[1] for c in cost[k]]) - baseline_memory
    lines.append(f"| {k} | {np.mean([x[0] for x in v]):.3f} | {np.mean([x[2] for x in v]):.1f} | "
                 f"{100 * np.mean([x[1] for x in v]):.2f} % | {np.mean([x[3] for x in v]):.0f} px | "
                 f"{np.mean([x[4] for x in v]):.1f} px | {params[k]:,} | {1000 * np.mean([c[0] for c in cost[k]]):.1f} ms | "
                 f"{memory:.0f} MiB |")
lines += ["", "Quality measures of (GD)^2 (Ahmed et al. 2020), also used by DeepGD and SmartGD (arrow: better direction):", "",
          "| method | " + " | ".join(f"{name} {'↑' if up else '↓'}" for name, (_, up) in MEASURES.items()) + " |",
          "|---|" + "---|" * len(MEASURES)]
for k, v in literature.items():
    lines.append(f"| {k} | " + " | ".join(f"{np.mean([x[c] for x in v]):.3f}" for c in range(len(MEASURES))) + " |")
torch.save({k: {name: P.cpu() for name, P in pos.items()} for k, pos in positions.items()}, OUT / "positions.pt")
(OUT / "results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# --- example drawings, rendered with our rasterizer in the 500 x 500 frame (node radius 4 px, edge width 1 px)
from data.output import Layout, save_png
RENDERS = OUT / "renders"; RENDERS.mkdir(parents=True, exist_ok=True)
for old in RENDERS.glob("*.png"):  # the sheet uses every file of the folder: keep only this run's examples
    old.unlink()
by_size = sorted(graphs, key=lambda g: g.num_nodes)
examples = [min(by_size, key=lambda g: abs(g.num_nodes - n)) for n in (30, 60, 100)]
for k, _ in methods:
    for g in examples:
        save_png(Layout(positions[k][g.name], W, H), g, RENDERS / f"{k.replace(' ', '_')}_{g.name}.png",
                 mode="raster", node_color="#2563a8", edge_color="#4b5563")
from make_sheet import make_sheet
make_sheet(RENDERS, OUT / "sheet.png")
