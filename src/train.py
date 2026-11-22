import copy
import os
import random
from pathlib import Path

# Less GPU memory fragmentation (4 GB card); must be set before CUDA is initialized.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data.input import GraphInput, flip_pe_signs, load_dataset, redraw_noise, split_dataset
from model import LayoutModel
from loss import Loss, LossFn



def batch_loss(model: LayoutModel, batch: Batch, loss_fn: LossFn, augment: bool = False,
               iterations: int | None = None) -> torch.Tensor:
    """Mean loss over the graphs of a batch.

    The model sees the whole batch at once (disjoint union of the graphs), and so does the loss: every
    metric handles a PyG batch and returns the mean over its graphs (see metrics/batching.py). Same value
    as the former loop over the graphs, without one series of GPU calls per graph.
    augment: draw new noise and new positional-encoding signs for each graph (training only), on the
    whole batch at once, without splitting it into graphs.
    iterations: number of model iterations, the model's default if None.
    The batch is moved to the model's device (the DataLoader yields CPU batches).
    """
    batch = batch.to(next(model.parameters()).device)
    if augment:
        batch = flip_pe_signs(redraw_noise(batch))
    layout = model(batch, iterations)
    return loss_fn(layout, batch)[0]


def train_step(model: LayoutModel, batch: Batch, loss_fn: LossFn, optimizer: torch.optim.Optimizer) -> float:
    """One optimizer step. Two tricks from CoRe-GD, both kept after testing (docs/idees/melissa-archi.md):
    - a random number of iterations around model.iterations, so the model works for any number of rounds;
    - gradient clipping, so that one bad batch cannot make a huge update.
    """
    iterations = max(round(random.gauss(model.iterations, 1.0)), 1)
    optimizer.zero_grad()
    loss = batch_loss(model, batch, loss_fn, augment=True, iterations=iterations)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def validation_loss(model: LayoutModel, graphs: list[GraphInput], loss_fn: LossFn, batch_size: int = 32) -> float:
    """Mean loss per graph. Positional encodings are seen as they are (no sign flip)."""
    if not graphs:
        return 0.0

    model.eval()
    try:
        total = 0.0
        for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            total += float(batch_loss(model, batch, loss_fn).item()) * batch.num_graphs
        return total / len(graphs)
    finally:
        model.train()


def train(
    model: LayoutModel,
    graphs: list[GraphInput],
    loss_fn: LossFn,
    epochs: int,
    lr: float = 2e-4,
    log_every: int = 0,
    batch_size: int = 1,
    val_graphs: list[GraphInput] | None = None,
    patience: int | None = None,
) -> list[float]:
    """Train the model on a list of graphs for a number of epochs.

    The graphs are shuffled every epoch and grouped into batches of batch_size graphs (of mixed sizes);
    one optimizer step per batch, on the mean loss of its graphs.
    """
    if patience is not None and val_graphs is None:
        raise ValueError("patience requires val_graphs to be provided.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss: float | None = None
    epochs_without_improvement = 0

    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        total_weight = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False, unit="batch"):
            epoch_loss += train_step(model, batch, loss_fn, optimizer) * batch.num_graphs
            total_weight += batch.num_graphs

        epoch_loss /= total_weight if total_weight > 0 else 1
        history.append(epoch_loss)

        if val_graphs is not None:
            val_loss = validation_loss(model, val_graphs, loss_fn, batch_size)
            if best_val_loss is None or val_loss < best_val_loss:
                best_state = copy.deepcopy(model.state_dict())
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if log_every > 0 and epoch % log_every == 0:
                print(f"Epoch {epoch}: train_loss = {epoch_loss:.4f}, val_loss = {val_loss:.4f}")
        elif log_every > 0 and epoch % log_every == 0:
            print(f"Epoch {epoch}: loss = {epoch_loss:.4f}")

        if patience is not None and val_graphs is not None:
            if epochs_without_improvement >= patience:
                if best_state is not None:
                    model.load_state_dict(best_state)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history

def main(
    data_dir: Path,
    out_path: Path,
    epochs: int,
    lr: float,
    loss_fn: LossFn,
    width: int = 500,
    height: int = 500,
    batch_size: int = 1,
    patience: int | None = None,
    init_path: Path | None = None,
) -> None:
    """Train a model and save it to out_path, with the loss weights.
    init_path: start from this trained checkpoint instead of a new model (phase 2, fine-tuning)."""
    train_files, val_files, test_files = split_dataset(data_dir)
    graphs = load_dataset(data_dir)
    train_graphs = [graphs[path.stem] for path in train_files]
    val_graphs = [graphs[path.stem] for path in val_files]
    test_graphs = [graphs[path.stem] for path in test_files]

    if not train_graphs:
        raise ValueError(f"No training graphs found in {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    in_features = train_graphs[0].x.shape[1]
    if init_path is None:
        model = LayoutModel(in_features, hidden=64, width=width, height=height, margin=4.0).to(device)
    else:
        from infer import load_model  # here: infer is only needed for fine-tuning
        model = load_model(init_path).to(device)
        model.train()
    train(
        model,
        train_graphs,
        loss_fn=loss_fn,
        epochs=epochs,
        lr=lr,
        log_every=1,
        batch_size=batch_size,
        val_graphs=val_graphs,
        patience=patience,
    )

    if val_graphs:
        val_loss = validation_loss(model, val_graphs, loss_fn, batch_size)
        print(f"Validation loss: {val_loss:.4f}")
    if test_graphs:
        test_loss = validation_loss(model, test_graphs, loss_fn, batch_size)
        print(f"Test loss: {test_loss:.4f}")

    checkpoint = {
        "state_dict": {name: tensor.cpu() for name, tensor in model.state_dict().items()},
        "config": model.config,
        "loss_weights": dict(loss_fn.weights),  # to know which loss produced the model
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)


# Loss in two phases, chosen on the validation split (runs: 5 + 3 fine-tunings, see docs/jalon2):
# 1. the stress alone gives the structure of the drawing (pixel-scaled stress, from scratch);
# 2. a short fine-tuning adds the screen metrics, each at 3 % of its "balanced" weight (1 / its mean
#    value on the phase-1 drawings, DeepGD's Adaptive Weight), so each weighs about 3 % of the stress.
# Test (1000 graphs) vs phase 1 continued alone: stress +3.8 %, crossings -5 %, overlapping nodes -64 %,
# edges over nodes -15 %, better angles. Heavier weights (10 %, 30 %, 100 %) or all metrics from the
# start degrade the stress much more (from scratch at 100 %: stress x3, 2.4x more crossings).
# No screen-filling term (2 - dx/W - dy/H, tried then removed): the pixel-scaled stress already sets the size
# of the drawing. Validation, 3 seeds, fine-tuning with it at 3 %: same stress (0.0612), same size (457 px),
# crossings 28.6 vs 28.9 but not on every seed (noise).
PHASE_1_WEIGHTS = {"stress": 1}
# Phase 2 uses BOTH stresses. The scale-normalized one (shape only, at the best scale: what the benchmark
# measures, and what CoRe-GD trains on) improves the shape; the pixel-scaled one keeps the size. With the
# scale-normalized stress alone, the drawings shrank by 24 % (nothing held the size, and the crossing cost,
# in pixels, decreases when the drawing shrinks): the shape metrics improved, the drawings got cramped.
# Validation, with refinement (REFINE_WEIGHTS), vs phase 2 with the pixel-scaled stress alone: stress
# 0.0609 vs 0.0615, edges over nodes 0.04 vs 0.07, angles 0.651 vs 0.647, crossings 28.2 vs 27.9, same size
# (460 px, nearest node 45 px).
PHASE_2_WEIGHTS = {
    "stress": 1,
    "scaled_stress": 1,
    "node_overlaps": 0.334,
    "edge_overlaps": 0.0518,
    "edge_anlges": 10.6,
    "edge_crossings": 0.0131,
    "edge_node_overlaps": 0.0646,
}
# Loss of the refinement after the model (refine.py): the same, but with the pixel-scaled stress only.
# Validation, on the phase-2 model: with both stresses the stress is a bit lower (0.0604 vs 0.0609), but
# edges over nodes (0.12 vs 0.04) and angles (0.643 vs 0.651) get worse.
REFINE_WEIGHTS = {name: weight for name, weight in PHASE_2_WEIGHTS.items() if name != "scaled_stress"}

if __name__ == "__main__":
    data_dir = Path("data/rome")
    # phase 1: stress alone, from scratch (~40 min on an RTX 3060)
    main(data_dir, Path("runs/model_phase1.pt"), epochs=30, lr=2e-4, loss_fn=Loss(weights=PHASE_1_WEIGHTS),
         batch_size=4, patience=4)
    # phase 2: fine-tuning with the screen metrics (~25 min)
    main(data_dir, Path("runs/model.pt"), epochs=8, lr=1e-4, loss_fn=Loss(weights=PHASE_2_WEIGHTS),
         batch_size=4, patience=3, init_path=Path("runs/model_phase1.pt"))
