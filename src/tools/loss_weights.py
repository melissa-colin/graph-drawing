"""Choose the loss coefficients: measure every metric of loss.py, then print the weights given by the
loss weighting methods of the literature (graph drawing, multi-task learning, physics-informed networks),
with the details of each computation, so the methods can be compared and tried.

Run from the project root:  python src/tools/loss_weights.py
The settings are the constants at the top of the file (N_DRAWS, USE_MODEL, DETAILS, ...).
Every measure uses every graph of the training split.

Measures, on graphs of the TRAINING split:
1. Scale: value and gradient (w.r.t. the node positions) of each metric on several layout configurations
   (uniform random, compact random: N_DRAWS draws each; Kamada-Kawai, small Kamada-Kawai: deterministic),
   and the share of each metric in the loss when all weights are 1.
2. Speed: a short optimization, every metric divided by its initial value, to see which metric decreases
   fastest. The rate-based methods read its loss curves. Two modes:
   - default: direct optimization of the node positions (Adam), as (GD)^2 does, graph by graph;
   - USE_MODEL = True: training of a fresh LayoutModel with train.train_step (augmentations, random iterations,
     gradient clipping), the gradient goes through the GNN and the soft-argmax head as in a real training.
     One step = one pass over the descent graphs. Slower: a few minutes.

Most methods of the literature update the weights during training. Here every method is evaluated once,
on these measures: the printed weights are a starting point, not the exact training-time algorithm.
All weights are rescaled to sum to K, the number of implemented metrics.

References
- (GD)^2: Ahmed, De Luca, Devkota, Kobourov, Li, Graph Drawing via Gradient Descent, GD 2020;
  (SGD)^2: same authors, IEEE TVCG 2022. Weights chosen by hand, with hand-crafted weight schedules.
- DeepGD: Wang, Yen, Hu, Shen, IEEE CG&A 2021, section 4.4 (Adaptive Weight, SoftAdapt by Importance).
- Uncertainty weighting: Kendall, Gal, Cipolla, CVPR 2018.
- GradNorm: Chen, Badrinarayanan, Lee, Rabinovich, ICML 2018.
- MGDA: Sener, Koltun, Multi-Task Learning as Multi-Objective Optimization, NeurIPS 2018.
- DWA: Liu, Johns, Davison, End-to-End Multi-Task Learning with Attention, CVPR 2019.
- SoftAdapt: Heydari, Thompson, Mehmood, 2019.
- Learning Rate Annealing: Wang, Teng, Perdikaris, SIAM J. Sci. Comput. 2021 (PINNs).
- CoV weighting: Groenendijk, Karaoglu, Gevers, Mensink, WACV 2021.
- IMTL-L: Liu et al., Towards Impartial Multi-task Learning, ICLR 2021.
- ReLoBRaLo: Bischof, Kraus, Multi-Objective Loss Balancing for Physics-Informed Deep Learning, 2021.
- RLW: Lin et al., Reasonable Effectiveness of Random Weighting, TMLR 2022.
- Unitary scalarization: Kurin et al., In Defense of the Unitary Scalarization, NeurIPS 2022.
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/

from data.input import load_dataset, split_dataset
from data.output import Layout
from loss import edge_angles, edge_overlaps, node_overlaps, screen_fill, stress
from model import LayoutModel
from train import batch_loss, train_step

# Same keys as the metrics dict of Loss.forward, so the printed weights can be pasted into Loss(weights=...).
METRICS = {
    "screen_fill": screen_fill,
    "stress": stress,
    "node_overlaps": node_overlaps,
    "edge_overlaps": edge_overlaps,
    "edge_anlges": edge_angles,
}
# ----------------------------------------------------------------------------- settings
N_DRAWS = 1             # random drawings per graph and random configuration, random starts of the direct descent
USE_MODEL = False       # True: measure the speeds by training our LayoutModel instead of moving the positions
STEPS = 30 if USE_MODEL else 100   # descent steps (passes over the descent graphs with the model)
LR = 2e-4               # learning rate of the model descent (train.py default)
BATCH_SIZE = 4          # batch size of the model descent
REFERENCE = "stress"    # main loss for Learning Rate Annealing
IMPORTANCE = {}         # DeepGD importance factors, e.g. {"stress": 2.0}; missing metrics: 1
TEMPERATURE = 1.0       # softmax temperature (SoftAdapt, ReLoBRaLo, DeepGD SoftAdapt)
SEED = 0
DETAILS = False         # True: print every measure, formula, reference and Loss(weights=...) line

WIDTH = HEIGHT = 500
MARGIN = 4.0
EPS = 1e-12


# ----------------------------------------------------------------------------- layouts

def nx_graph(graph):
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    G.add_edges_from(graph.edge_index.t().tolist())
    return G


def fit_box(positions, size):
    """Rescale uniformly into a size x size box centered in the image."""
    positions = positions - positions.mean(0)
    positions = positions * (size / 2 - MARGIN) / positions.abs().max().clamp_min(EPS)
    return positions + torch.tensor([WIDTH / 2, HEIGHT / 2])


def random_box(n, size, rng):
    low = torch.tensor([WIDTH / 2 - size / 2 + MARGIN, HEIGHT / 2 - size / 2 + MARGIN])
    return low + torch.rand(n, 2, generator=rng) * (size - 2 * MARGIN)


def configurations(graph, rng, draws):
    """Layouts on which the metrics are measured: bad, cramped and good drawings.
    Yields (configuration name, positions); the two random configurations are drawn `draws` times."""
    for _ in range(draws):
        yield "uniform", random_box(graph.num_nodes, WIDTH, rng)
        yield "compact", random_box(graph.num_nodes, 60, rng)
    kk = nx.kamada_kawai_layout(nx_graph(graph))
    kk = torch.tensor(np.array([kk[i] for i in range(graph.num_nodes)]), dtype=torch.float32)
    yield "kamada_kawai", fit_box(kk, WIDTH)
    yield "kk_small", fit_box(kk, 100)


# ----------------------------------------------------------------------------- measures

def evaluate(metric, positions, graph):
    """(value, gradient w.r.t. the positions) of one metric, or None if the metric is not implemented."""
    positions = positions.clone().requires_grad_(True)
    value = metric(Layout(positions, WIDTH, HEIGHT), graph)
    if not torch.is_tensor(value):
        return None
    if value.requires_grad:
        value.backward()
    grad = positions.grad if positions.grad is not None else torch.zeros_like(positions)
    return value.item(), grad.flatten()


def descent(graph, active, steps, rng):
    """Direct optimization of the positions, every metric divided by its initial value (all start at 1).
    Returns the curve of each metric: {name: [L(0), ..., L(steps)]}."""
    positions = random_box(graph.num_nodes, WIDTH, rng).requires_grad_(True)
    optimizer = torch.optim.Adam([positions], lr=2.0)
    initial = {name: max(evaluate(METRICS[name], positions.detach(), graph)[0], EPS) for name in active}
    curves = {name: [] for name in active}
    for _ in range(steps + 1):
        layout = Layout(positions, WIDTH, HEIGHT)
        values = {name: METRICS[name](layout, graph) for name in active}
        for name, value in values.items():
            curves[name].append(value.item())
        total = sum(value / initial[name] for name, value in values.items())
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        with torch.no_grad():
            positions[:, 0].clamp_(MARGIN, WIDTH - MARGIN)
            positions[:, 1].clamp_(MARGIN, HEIGHT - MARGIN)
    return curves


def model_descent(graphs, active, steps, lr, batch_size):
    """Training of a fresh LayoutModel on the descent graphs, every metric divided by its initial value
    (measured on the untrained model). Returns, per graph, the curve of each metric:
    {graph name: {name: [L(0), ..., L(steps)]}}, one point per pass over the graphs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayoutModel(graphs[0].x.shape[1], hidden=64, width=WIDTH, height=HEIGHT, margin=MARGIN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    last = {name: {} for name in active}     # last value seen for each metric and graph
    initial = {name: {} for name in active}

    def loss_fn(layout, batch):
        # train.batch_loss gives the whole batch: the metrics are computed graph by graph here, because
        # each graph has its own initial value to divide by (slower than the training loss, fine for a tool)
        ptr = batch.ptr.tolist()
        totals = []
        for k, graph in enumerate(batch.to_data_list()):
            part = Layout(layout.positions[ptr[k]:ptr[k + 1]], layout.width, layout.height)
            values = {name: METRICS[name](part, graph) for name in active}
            for name, value in values.items():
                last[name][graph.name] = value.item()
            if not initial[active[0]]:        # first pass: only record the initial values
                totals.append(sum(values.values()))
            else:
                totals.append(sum(value / max(initial[name][graph.name], EPS) for name, value in values.items()))
        return torch.stack(totals).mean(), {}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_loss(model, batch, loss_fn)
    model.train()
    for name in active:
        initial[name] = dict(last[name])
    curves = {g.name: {name: [initial[name][g.name]] for name in active} for g in graphs}
    with tqdm(total=steps * len(loader), desc="Descente (modèle)", unit="batch", leave=True) as bar:
        for step in range(steps):
            for batch in loader:
                train_step(model, batch, loss_fn, optimizer)
                bar.update()
            bar.set_postfix_str(f"passage {step + 1}/{steps}")
            for g in graphs:
                for name in active:
                    curves[g.name][name].append(last[name][g.name])
    return curves


def min_norm_weights(grads):
    """MGDA: the convex combination of the gradients with the smallest norm (Frank-Wolfe, Sener & Koltun).
    grads: (K, D). Returns alpha (K,), on the simplex."""
    G = grads.double()
    gram = G @ G.T
    alpha = torch.full((len(G),), 1.0 / len(G), dtype=torch.float64)
    for _ in range(250):
        t = int(torch.argmin(gram @ alpha))           # vertex that most decreases the norm
        theta, theta_t = alpha @ G, G[t]
        diff = theta - theta_t
        if diff.norm() < 1e-12:
            break
        # min over gamma in [0, 1] of ||(1 - gamma) theta + gamma theta_t||^2
        gamma = float(torch.clamp((theta @ diff) / (diff @ diff), 0.0, 1.0))
        if gamma < 1e-9:
            break
        alpha = (1 - gamma) * alpha
        alpha[t] += gamma
    return alpha.float()


def softmax(values, temperature):
    v = np.array(values) / temperature
    v = np.exp(v - v.max())  # minus the max: same result, no overflow
    return v / v.sum()


# ----------------------------------------------------------------------------- printing

def normalized(raw):
    """Rescale so the weights sum to the number of metrics."""
    total = sum(raw.values())
    return {name: len(raw) * w / total for name, w in raw.items()}


def table(title, rows, columns, fmt="{:>14.4g}"):
    print(f"\n{title}")
    print(f"  {'':<16}" + "".join(f"{c:>14}" for c in columns))
    for name, values in rows.items():
        print(f"  {name:<16}" + "".join(fmt.format(v) for v in values))


RESULTS = []      # (family, short name, weights) of every method, for the summary table
FAMILY = [""]
SHORT_NAMES = [   # title prefix -> name in the summary
    ("Uniform", "Uniforme"), ("RLW", "Aléatoire (RLW)"), ("DeepGD, Adaptive", "DeepGD adaptatif"),
    ("DeepGD, SoftAdapt", "DeepGD SoftAdapt"), ("Inverse magnitude", "Inverse moyenne"),
    ("Inverse median", "Inverse médiane"), ("Uncertainty", "Incertitude"), ("CoV", "CoV"),
    ("Gradient norm", "Équilibre gradients"), ("GradNorm", "GradNorm"), ("Learning Rate", "LR Annealing"),
    ("MGDA", "MGDA"), ("DWA", "DWA"), ("SoftAdapt", "SoftAdapt"), ("ReLoBRaLo", "ReLoBRaLo"),
]
FAMILY_NAMES = {"A": "Références", "B": "Dessin de graphes", "C": "Échelle des losses",
                "D": "Gradients", "E": "Vitesse de descente"}


def family(title):
    FAMILY[0] = FAMILY_NAMES[title[0]]
    if DETAILS:
        print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def show_method(title, reference, formula, details, weights):
    short = next(name for prefix, name in SHORT_NAMES if title.startswith(prefix))
    RESULTS.append((FAMILY[0], short, weights))
    if not DETAILS:
        return
    print(f"\n--- {title}   [{reference}]")
    for line in formula.split("\n"):
        print(f"    {line}")
    for name in weights:
        print(f"    {name:<16} {details.get(name, ''):<58} -> w = {weights[name]:.4g}")
    pasted = ", ".join(f'"{name}": {weights[name]:.4g}' for name in weights)
    print(f"    Loss(weights={{{pasted}}})")


def weight_text(w):
    return f"{w:.2f}" if w >= 0.01 else f"{w:.0e}"


def summary(active, mean, gnorm, remaining):
    """Short summary, in French: what each metric weighs, the weights of every method, a few findings."""
    bold = (lambda t: f"\033[1m{t}\033[0m") if sys.stdout.isatty() else (lambda t: t)
    total = sum(mean.values())
    share = {n: 100 * mean[n] / max(total, EPS) for n in active}
    col = max(14, *(len(n) + 2 for n in active))

    print(bold("\nLes métriques, tous poids à 1"))
    print(f"  {'':<16}{'valeur moy.':>14}{'part de la loss':>17}{'gradient moy.':>15}{'reste après descente':>22}")
    for n in active:
        print(f"  {n:<16}{mean[n]:>14.3g}{share[n]:>16.1f}%{gnorm[n]:>15.3g}{100 * remaining[n]:>21.0f}%")

    print(bold("\nÀ retenir"))
    top = max(active, key=share.get)
    if share[top] > 90:
        print(f"  - {top} fait {share[top]:.0f} % de la loss avec des poids à 1 : il écrase les autres.")
    weak = min(active, key=gnorm.get)
    if gnorm[weak] < 1e-2 * max(gnorm.values()):
        ratio = f"{max(gnorm.values()) / max(gnorm[weak], EPS):,.0f}".replace(",", " ")
        print(f"  - {weak} a un gradient {ratio} fois plus faible que le plus fort : il pousse à peine les nœuds.")
    fast, slow = min(active, key=remaining.get), max(active, key=remaining.get)
    print(f"  - {fast} diminue le plus vite (il en reste {100 * remaining[fast]:.0f} %), {slow} le moins vite ({100 * remaining[slow]:.0f} %).")

    print(bold("\nPoids proposés par méthode (somme = nombre de métriques)"))
    print(f"  {'':<22}{'':<21}" + "".join(f"{n:>{col}}" for n in active))
    previous = None
    for fam, short, weights in RESULTS:
        label = fam if fam != previous else ""
        previous = fam
        print(f"  {label:<22}{short:<21}" + "".join(f"{weight_text(weights[n]):>{col}}" for n in active))
    print(bold("\nPar où commencer"))
    print("  1. Inverse moyenne (= DeepGD adaptatif) : le plus simple, déjà utilisé en dessin de graphes ;")
    print("     chaque terme vaut environ 1 au départ. Point de départ conseillé.")
    print("  2. Incertitude (Kendall et al. 2018) : la plus citée en apprentissage multitâche. Elle s'apprend")
    print("     pendant l'entraînement (un paramètre par métrique) ; l'estimation ci-dessus n'est qu'indicative.")
    print("  3. GradNorm ou DWA : si une métrique reste à la traîne pendant l'entraînement.")
    if gnorm[weak] < 1e-2 * max(gnorm.values()):
        print(f"  À éviter pour l'instant : les méthodes de la famille Gradients, qui donnent presque tout le poids")
        print(f"  à {weak} parce que son gradient est quasi nul.")
    print("  Rappel : deux grandes comparaisons (Kurin et al. 2022, Xin et al. 2022) trouvent qu'aucune méthode")
    print("  adaptative ne bat nettement des poids fixes bien réglés : commencer simple.")
    print("\n  Détail des calculs, formules, références et lignes Loss(weights=...) à copier : DETAILS = True en haut du fichier.")


# ----------------------------------------------------------------------------- main

def main():
    rng = torch.Generator().manual_seed(SEED)
    data_dir = Path("data/rome")
    print("Chargement des graphes et du découpage train/val/test...")
    graphs = load_dataset(data_dir)
    train_files, _, _ = split_dataset(data_dir)
    train_graphs = [graphs[p.stem] for p in train_files]
    sample = train_graphs
    mode = "entraînement de notre modèle" if USE_MODEL else "optimisation directe des positions"
    print(f"{len(sample)} graphes d'entraînement ({min(g.num_nodes for g in sample)} à {max(g.num_nodes for g in sample)} nœuds), "
          f"{N_DRAWS} tirage(s) aléatoire(s) par graphe, vitesse mesurée par {mode}.")

    # ------------------------------------------------------------------ 1. scale
    values, grad_norm, grad_max, grad_mean = ({n: {} for n in METRICS} for _ in range(4))
    mgda_alphas, mgda_skipped = [], 0
    missing = set()
    for graph in tqdm(sample, desc="Mesures d'échelle", unit="graphe", leave=True):
        for config, positions in configurations(graph, rng, N_DRAWS):
            grads = {}
            for name, metric in METRICS.items():
                result = evaluate(metric, positions, graph)
                if result is None:
                    missing.add(name)
                    continue
                value, grad = result
                grads[name] = grad
                values[name].setdefault(config, []).append(value)
                grad_norm[name].setdefault(config, []).append(grad.norm().item())
                grad_max[name].setdefault(config, []).append(grad.abs().max().item())
                grad_mean[name].setdefault(config, []).append(grad.abs().mean().item())
            stacked = torch.stack(list(grads.values()))
            if (stacked.norm(dim=1) > 0).all():  # MGDA is meaningless when one gradient is already 0
                mgda_alphas.append(min_norm_weights(stacked).numpy())
            else:
                mgda_skipped += 1
    active = [name for name in METRICS if name not in missing]
    K = len(active)
    if missing:
        print(f"Pas encore implémentées, ignorées : {', '.join(sorted(missing))}.")
    reference = REFERENCE if REFERENCE in active else active[0]
    configs = list(values[active[0]])

    if DETAILS:
        table("1a. Mean value per configuration", {n: [np.mean(values[n][c]) for c in configs] for n in active}, configs)
        table("1b. Mean gradient norm ||dL/dpositions|| per configuration",
              {n: [np.mean(grad_norm[n][c]) for c in configs] for n in active}, configs)
    shares = {}
    for c in configs:
        total = sum(np.mean(values[n][c]) for n in active)
        for n in active:
            shares.setdefault(n, []).append(100 * np.mean(values[n][c]) / max(total, EPS))
    if DETAILS:
        table("1c. Share of each metric in the loss with all weights at 1 (%)", shares, configs, fmt="{:>13.1f}%")

    def pooled(stat, n):
        return np.concatenate([stat[n][c] for c in configs])
    mean = {n: float(pooled(values, n).mean()) for n in active}
    median = {n: float(np.median(pooled(values, n))) for n in active}
    std = {n: float(pooled(values, n).std()) for n in active}
    gnorm = {n: float(pooled(grad_norm, n).mean()) for n in active}
    gmax = {n: float(pooled(grad_max, n).mean()) for n in active}
    gmean = {n: float(pooled(grad_mean, n).mean()) for n in active}

    # ------------------------------------------------------------------ 2. speed
    descent_graphs = sample
    remaining, recent, delta_rel = ({n: [] for n in active} for _ in range(3))
    window = max(STEPS // 10, 1)
    if USE_MODEL:
        all_curves = list(model_descent(descent_graphs, active, STEPS, LR, BATCH_SIZE).values())
    else:
        starts = [graph for graph in descent_graphs for _ in range(N_DRAWS)]
        all_curves = [descent(graph, active, STEPS, rng)
                      for graph in tqdm(starts, desc="Descente (positions)", unit="départ", leave=True)]
    for curves in all_curves:
        for n in active:
            curve = curves[n]
            if curve[0] <= EPS:  # already satisfied at the start (e.g. no overlap): no rate to measure
                continue
            floor = 1e-3 * curve[0]  # a metric that reaches 0 must not make the ratios explode
            remaining[n].append(curve[-1] / curve[0])
            recent[n].append((curve[-1] + floor) / (curve[-1 - window] + floor))
            delta_rel[n].append((curve[-1] - curve[-1 - window]) / (curve[-1 - window] + floor))
    remaining = {n: float(np.mean(remaining[n])) if remaining[n] else 1.0 for n in active}
    recent = {n: float(np.mean(recent[n])) if recent[n] else 1.0 for n in active}
    delta_rel = {n: float(np.mean(delta_rel[n])) if delta_rel[n] else 0.0 for n in active}
    mode = (f"training of a fresh LayoutModel ({STEPS} passes, lr {LR:g})" if USE_MODEL
            else f"direct optimization of the positions ({STEPS} Adam steps, {N_DRAWS} random start(s) per graph)")
    if DETAILS:
        table(f"2. Descent on {len(descent_graphs)} graphs: {mode}, every metric divided by its initial value",
              {n: [remaining[n], recent[n]] for n in active}, ["L(T)/L(0)", f"last {window} st."])
        print("  L(T)/L(0): fraction left at the end (small = decreased fast); last steps: ratio over the last steps (1 = stalled).")

    importance = {n: IMPORTANCE.get(n, 1.0) for n in active}

    if DETAILS:
        print(f"\n3. Weights by method, rescaled to sum to K = {K}")

    # ------------------------------------------------------------------ baselines
    family("A. Baselines")
    show_method("Uniform, unitary scalarization", "Kurin et al. 2022",
                "w_k = 1. With a tuned optimizer, often as good as the adaptive methods.",
                {}, normalized({n: 1.0 for n in active}))
    draw = softmax(np.random.default_rng(SEED).normal(size=K), 1.0) * K
    show_method("RLW, Random Loss Weighting", "Lin et al. 2022",
                "No fixed weights: at every step, w = K softmax(z), z ~ N(0, 1). One draw, as an example:",
                {}, dict(zip(active, draw)))

    # ------------------------------------------------------------------ graph drawing
    family("B. Graph drawing")
    if DETAILS:
        print("\n--- (GD)^2 and (SGD)^2   [Ahmed et al. 2020, 2022]")
        print("    Weights chosen by hand for each combination of criteria, with hand-crafted schedules")
        print("    (weights that change during the optimization); no formula to compute them.")
    raw = {n: importance[n] / max(mean[n], EPS) for n in active}
    show_method("DeepGD, Adaptive Weight", "Wang et al. 2021, eq. 12",
                "alpha_k = (gamma_k / L_k) / sum_l (gamma_l / L_l), gamma = importance factor (IMPORTANCE),\n"
                "L_k = average loss of the previous epoch (here: mean over the configurations).",
                {n: f"gamma = {importance[n]:g}, L = {mean[n]:.4g}" for n in active}, normalized(raw))
    beta = 1.0 / TEMPERATURE
    s = np.array([delta_rel[n] for n in active])
    s = s / max(np.abs(s).sum(), EPS)  # normalize_L1 of the relative changes
    raw = {n: importance[n] / max(mean[n], EPS) * np.exp(beta * s_k) for n, s_k in zip(active, s)}
    show_method("DeepGD, SoftAdapt by Importance", "Wang et al. 2021, eq. 13-15",
                "alpha_k proportional to (gamma_k / L_k) exp(beta s_k), s = L1-normalized relative change of each loss\n"
                f"over the last {window} descent steps, beta = 1 / temperature = {beta:g}.",
                {n: f"L = {mean[n]:.4g}, s = {s_k:.3g}" for n, s_k in zip(active, s)}, normalized(raw))

    # ------------------------------------------------------------------ scale normalization
    family("C. Scale normalization (loss values only)")
    show_method("Inverse magnitude", "IMTL-L, Liu et al. 2021: at equilibrium w_k L_k = constant",
                "w_k proportional to 1 / mean(L_k): every term contributes about the same value.",
                {n: f"mean(L) = {mean[n]:.4g}" for n in active}, normalized({n: 1 / max(mean[n], EPS) for n in active}))
    show_method("Inverse median", "same, robust to extreme configurations",
                "w_k proportional to 1 / median(L_k)",
                {n: f"median(L) = {median[n]:.4g}" for n in active}, normalized({n: 1 / max(median[n], EPS) for n in active}))
    show_method("Uncertainty weighting, static estimate", "Kendall et al. 2018",
                "w_k = 1 / (2 sigma_k^2); sigma is learned during training, here estimated as the std of L_k\n"
                "across configurations.",
                {n: f"sigma = {std[n]:.4g}" for n in active}, normalized({n: 1 / (2 * max(std[n] ** 2, EPS)) for n in active}))
    cov = {n: std[n] / max(mean[n], EPS) for n in active}
    show_method("CoV weighting, static estimate", "Groenendijk et al. 2021",
                "w_k proportional to sigma_k / mu_k (coefficient of variation); they compute it on loss ratios during\n"
                "training, here across configurations.",
                {n: f"sigma/mu = {cov[n]:.4g}" for n in active}, normalized(cov))

    # ------------------------------------------------------------------ gradients
    family("D. Gradient-based")
    show_method("Gradient norm balance", "the balance targeted by GradNorm, Chen et al. 2018",
                "w_k proportional to 1 / mean ||grad L_k||: every term pushes the nodes with the same force.",
                {n: f"mean ||grad L|| = {gnorm[n]:.4g}" for n in active}, normalized({n: 1 / max(gnorm[n], EPS) for n in active}))
    alpha = 1.5
    rate = {n: remaining[n] / np.mean(list(remaining.values())) for n in active}
    g_bar = np.mean(list(gnorm.values()))
    raw = {n: g_bar / max(gnorm[n], EPS) * rate[n] ** alpha for n in active}
    show_method("GradNorm, one-shot", "Chen et al. 2018",
                f"w_k such that ||grad (w_k L_k)|| = mean(G) * r_k^alpha, r_k = (L_k(T)/L_k(0)) / mean, alpha = {alpha}:\n"
                "w_k = mean(G) / G_k * r_k^alpha (metrics that decreased slowly get more weight).",
                {n: f"G = {gnorm[n]:.3g}, r = {rate[n]:.3g}" for n in active}, normalized(raw))
    ref = reference
    raw = {n: gmax[ref] / max(gmean[n], EPS) if n != ref else 1.0 for n in active}
    show_method(f"Learning Rate Annealing (reference loss: {ref}, REFERENCE)", "Wang, Teng, Perdikaris 2021",
                "lambda_k = max |grad L_ref| / mean |grad L_k|, lambda_ref = 1\n"
                "(during training: moving average lambda = 0.9 lambda_old + 0.1 lambda_new).",
                {n: f"max|grad L_ref| = {gmax[ref]:.3g}, mean|grad L| = {gmean[n]:.3g}" for n in active}, normalized(raw))
    if mgda_alphas:
        mgda = np.mean(mgda_alphas, axis=0)
        show_method("MGDA, min-norm combination of the gradients", "Sener & Koltun 2018",
                    "alpha = argmin over the simplex of ||sum_k alpha_k grad L_k|| (Frank-Wolfe), averaged over\n"
                    f"{len(mgda_alphas)} graph x configuration samples ({mgda_skipped} skipped: one gradient already 0).",
                    {n: f"mean alpha = {a:.3g}" for n, a in zip(active, mgda)}, normalized(dict(zip(active, mgda))))

    # ------------------------------------------------------------------ rates of decrease
    family("E. Rate of decrease (loss curves of the descent)")
    tau = 2.0
    w = softmax([recent[n] for n in active], tau) * K
    show_method("DWA, Dynamic Weight Average", "Liu et al. 2019",
                f"w_k = K exp(r_k / T) / sum_j exp(r_j / T), r_k = L_k(t) / L_k(t-1) (here over the last {window} steps), T = {tau}",
                {n: f"r = {recent[n]:.4g}" for n in active}, normalized(dict(zip(active, w))))
    w = softmax([delta_rel[n] for n in active], TEMPERATURE) * K
    show_method("SoftAdapt (normalized variant)", "Heydari et al. 2019",
                f"w_k = softmax(s_k / T), s_k = (L_k(t) - L_k(t-1)) / L_k(t-1) over the last {window} steps, T = {TEMPERATURE:g}",
                {n: f"s = {delta_rel[n]:.4g}" for n in active}, normalized(dict(zip(active, w))))
    w = softmax([remaining[n] for n in active], TEMPERATURE) * K
    show_method("ReLoBRaLo, lookback to the start", "Bischof & Kraus 2021, eq. 11",
                f"lambda_k(t, 0) = K exp(L_k(t) / (T L_k(0))) / sum_j exp(L_j(t) / (T L_j(0))), T = {TEMPERATURE:g}\n"
                "(during training: random lookback and exponential decay between this and the previous step).",
                {n: f"L(T)/L(0) = {remaining[n]:.4g}" for n in active}, normalized(dict(zip(active, w))))

    if DETAILS:
        print("\nThe adaptive methods update the weights during training; the values above are a one-shot")
        print("estimate on fixed measures, to compare the methods and pick a starting point.")
    summary(active, mean, gnorm, remaining)


if __name__ == "__main__":
    main()
